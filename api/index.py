import os
import re
import json
import datetime
import traceback
import requests
from flask import Flask, request, Response
from supabase import create_client, Client as SupabaseClient
from groq import Groq
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Configuration — read from environment (never hardcoded)
# ---------------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
META_PHONE_NUMBER_ID = os.getenv("META_PHONE_NUMBER_ID")
META_WHATSAPP_BUSINESS_ACCOUNT_ID = os.getenv("META_WHATSAPP_BUSINESS_ACCOUNT_ID") or os.getenv("META_WABA_ID")
META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN")

META_GRAPH_API_VERSION = "v21.0"

# Default model aligned with remote; override via GROQ_MODEL env var
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")

# Log safe masked configuration (never log full keys/tokens)
logger.info(
    "[META CONFIG] phone_number_id=%s*** waba_id=%s*** webhook_path=/webhook/whatsapp",
    str(META_PHONE_NUMBER_ID)[:4] if META_PHONE_NUMBER_ID else "none",
    str(META_WHATSAPP_BUSINESS_ACCOUNT_ID)[:4] if META_WHATSAPP_BUSINESS_ACCOUNT_ID else "none"
)

# ---------------------------------------------------------------------------
# Lazy client initialisation
# Clients are created on first use to prevent startup crashes from killing
# all routes when an env var is missing.
# ---------------------------------------------------------------------------
_supabase_client = None
_groq_client = None


def get_supabase():
    global _supabase_client
    if _supabase_client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise RuntimeError("SUPABASE_URL or SUPABASE_KEY env var is missing")
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase_client


def get_groq():
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY env var is missing")
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


# ---------------------------------------------------------------------------
# Response sanitizer — strips <think>...</think> blocks (Requirement #2)
# These blocks MUST NEVER reach WhatsApp or be stored as the assistant reply.
# ---------------------------------------------------------------------------
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def sanitize_reply(text):
    """Strip any <think>...</think> reasoning blocks from the LLM output."""
    if not text:
        return ""
    cleaned = _THINK_RE.sub("", text).strip()
    return cleaned.strip()


# ---------------------------------------------------------------------------
# WhatsApp message length handling (Requirement — 4096 char limit)
# ---------------------------------------------------------------------------
WHATSAPP_MAX_MESSAGE_LEN = 4096


def cap_whatsapp_reply(text):
    """Ensure text fits WhatsApp's 4096-char limit.

    Tries to truncate at a natural break point (last newline within limit,
    then last space). Falls back to hard cut if no safe point exists.
    Returns (text, truncated_flag).
    """
    if not text:
        return text, False

    if len(text) <= WHATSAPP_MAX_MESSAGE_LEN:
        return text, False

    suffix = "\n\n... (truncated for WhatsApp)"
    limit = WHATSAPP_MAX_MESSAGE_LEN - len(suffix)

    truncated = text[:limit]
    nl_pos    = truncated.rfind("\n")
    space_pos = truncated.rfind(" ")

    if nl_pos > 0:
        truncated = truncated[:nl_pos]
    elif space_pos > 0:
        truncated = truncated[:space_pos]

    return truncated + suffix, True


# ---------------------------------------------------------------------------
# Validation helpers (Requirement #7)
# ---------------------------------------------------------------------------

def _resolve_project(supabase, project_name):
    """
    Resolve a project name (fuzzy match) to (project_id, canonical_name).
    Raises ValueError with a user-friendly message if:
      - project_name is empty
      - no project matches
      - multiple projects match (ambiguous)
    """
    if not project_name or not project_name.strip():
        raise ValueError("project_name is required but was empty.")
    res = supabase.table("projects").select("id, project_name").ilike(
        "project_name", f"%{project_name.strip()}%"
    ).execute()
    rows = res.data or []
    if len(rows) == 0:
        raise ValueError(
            f"No project found matching '{project_name}'. Check the name and try again."
        )
    if len(rows) > 1:
        names = ", ".join(r["project_name"] for r in rows)
        raise ValueError(
            f"Ambiguous project name '{project_name}' matched multiple projects: {names}. "
            "Please be more specific."
        )
    return rows[0]["id"], rows[0]["project_name"]


def _validate_date(value, field_name="date"):
    """Parse and return an ISO-8601 date string. Raises ValueError on failure."""
    try:
        parsed = datetime.date.fromisoformat(str(value).strip())
        return str(parsed)
    except (ValueError, TypeError):
        raise ValueError(
            f"'{field_name}' must be a valid ISO-8601 date (YYYY-MM-DD), got: '{value}'"
        )


def _validate_non_negative_number(value, field_name):
    """Cast value to float and ensure it is >= 0. Raises ValueError on failure."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{field_name}' must be a number, got: '{value}'")
    if num < 0:
        raise ValueError(f"'{field_name}' must be >= 0, got: {num}")
    return num


# ---------------------------------------------------------------------------
# Tool schemas
# Requirements: #4 read tools, #5 write tools, #6 dynamic attributes
# ---------------------------------------------------------------------------
TOOLS = [
    # ── READ TOOLS ──────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_projects",
            "description": (
                "Fetch the list of construction projects from the database "
                "(project name, location, status). Use whenever the user asks about "
                "active/ongoing projects, project list, or what projects exist."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Optional status filter e.g. 'Active'. Omit to return all."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_project_summary",
            "description": (
                "Fetch overall progress totals, total expenses, and equipment stats "
                "for a given project name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project_name": {
                        "type": "string",
                        "description": "The name of the project."
                    }
                },
                "required": ["project_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_daily_logs",
            "description": (
                "Get daily site progress logs (trenching, pipe laid, backfilling metres, notes) "
                "for a project, ordered by date descending."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project_name": {
                        "type": "string",
                        "description": "Name of the project (e.g. 'Pipeline Alpha')."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of log entries to return (default 5, max 50).",
                        "minimum": 1,
                        "maximum": 50
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_expenses",
            "description": (
                "Get site expenses (category, amount, payment mode, vendor/recipient, date, notes). "
                "Can optionally filter by project name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project_name": {
                        "type": "string",
                        "description": "Optional project name filter."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of expenses to return (default 5, max 50).",
                        "minimum": 1,
                        "maximum": 50
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_equipment_logs",
            "description": (
                "Get equipment usage logs (equipment name, hours operated, fuel consumed, "
                "operator name, notes, log date) for a project."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project_name": {
                        "type": "string",
                        "description": "Name of the project to fetch equipment logs for."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of equipment log entries to return (default 5, max 50).",
                        "minimum": 1,
                        "maximum": 50
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_project_attributes",
            "description": (
                "Fetch flexible project-specific attributes from the project_attributes table "
                "(e.g. workers_present, weather, soil_type, supervisor_name, machinery_count). "
                "Use when the user asks about custom/extra details for a project."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project_name": {
                        "type": "string",
                        "description": "Name of the project whose attributes to retrieve."
                    },
                    "attribute_name": {
                        "type": "string",
                        "description": "Optional: filter by a specific attribute name."
                    }
                },
                "required": ["project_name"]
            }
        }
    },
    # ── WRITE TOOLS ─────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "create_daily_log",
            "description": (
                "Create a new daily site progress log entry for a project. "
                "Use when the user wants to log today's (or a specific date's) trenching, "
                "pipe-laying, backfilling progress, or site notes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project_name": {
                        "type": "string",
                        "description": "Name of the project (resolved to project_id internally)."
                    },
                    "log_date": {
                        "type": "string",
                        "description": "Date of the log in YYYY-MM-DD format."
                    },
                    "trenching_meters": {
                        "type": "number",
                        "description": "Metres of trenching completed (optional, >= 0)."
                    },
                    "pipe_laid_meters": {
                        "type": "number",
                        "description": "Metres of pipe laid (optional, >= 0)."
                    },
                    "backfilling_meters": {
                        "type": "number",
                        "description": "Metres of backfilling completed (optional, >= 0)."
                    },
                    "raw_notes": {
                        "type": "string",
                        "description": "Free-text site notes (optional)."
                    }
                },
                "required": ["project_name", "log_date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_expense",
            "description": (
                "Record a new expense entry for a project. "
                "Use when the user wants to log a payment, purchase, or cost."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project_name": {
                        "type": "string",
                        "description": "Name of the project."
                    },
                    "expense_date": {
                        "type": "string",
                        "description": "Date of the expense in YYYY-MM-DD format."
                    },
                    "category": {
                        "type": "string",
                        "description": "Expense category (e.g. 'Labour', 'Fuel', 'Materials')."
                    },
                    "amount": {
                        "type": "number",
                        "description": "Amount in INR (must be >= 0)."
                    },
                    "payment_mode": {
                        "type": "string",
                        "description": "Payment mode (e.g. 'Cash', 'UPI', 'Bank Transfer')."
                    },
                    "vendor_or_recipient": {
                        "type": "string",
                        "description": "Vendor name or recipient (optional)."
                    },
                    "notes": {
                        "type": "string",
                        "description": "Additional notes (optional)."
                    }
                },
                "required": ["project_name", "expense_date", "category", "amount", "payment_mode"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_equipment_log",
            "description": (
                "Record a new equipment usage log entry for a project. "
                "Use when the user wants to log machinery/equipment usage for a day."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project_name": {
                        "type": "string",
                        "description": "Name of the project."
                    },
                    "log_date": {
                        "type": "string",
                        "description": "Date of the log in YYYY-MM-DD format."
                    },
                    "equipment_name": {
                        "type": "string",
                        "description": "Name/type of the equipment (e.g. 'Excavator', 'Compressor')."
                    },
                    "hours_operated": {
                        "type": "number",
                        "description": "Hours the equipment was operated (optional, >= 0)."
                    },
                    "fuel_consumed_liters": {
                        "type": "number",
                        "description": "Fuel consumed in litres (optional, >= 0)."
                    },
                    "operator_name": {
                        "type": "string",
                        "description": "Name of the operator (optional)."
                    },
                    "notes": {
                        "type": "string",
                        "description": "Additional notes (optional)."
                    }
                },
                "required": ["project_name", "log_date", "equipment_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_project_attribute",
            "description": (
                "Create or update a flexible project-specific attribute in the project_attributes table. "
                "Use for ANY custom/extra attribute that does not fit the standard columns: "
                "e.g. workers_present, weather, soil_type, machinery_count, supervisor_name, etc. "
                "Attributes belong to projects, NOT to database columns. "
                "If the attribute already exists for this project it is updated; "
                "if it does not exist it is created. Never creates duplicates for the same project."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project_name": {
                        "type": "string",
                        "description": "Name of the project."
                    },
                    "attribute_name": {
                        "type": "string",
                        "description": (
                            "Name of the attribute (snake_case preferred, e.g. 'workers_present', "
                            "'weather', 'soil_type'). Use exactly what the user specifies; "
                            "do NOT hardcode or invent attribute names."
                        )
                    },
                    "attribute_value": {
                        "type": "string",
                        "description": (
                            "The value to set (always stored as text). "
                            "E.g. '25' for workers_present, 'Sunny' for weather, 'Clay' for soil_type."
                        )
                    },
                    "attribute_type": {
                        "type": "string",
                        "description": (
                            "Data type hint: 'numeric', 'text', or 'date'. "
                            "Infer from context: numbers -> 'numeric', dates -> 'date', else 'text'."
                        ),
                        "enum": ["text", "numeric", "date"]
                    }
                },
                "required": ["project_name", "attribute_name", "attribute_value"]
            }
        }
    }
]


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------
def execute_tool(name, args):
    supabase = get_supabase()
    logger.info("[WRITE DEBUG] tool_name=%s", name)
    logger.info("[WRITE DEBUG] tool_arguments=%s", args)
    logger.info("[TOOL DEBUG] called=true name=%s", name)

    # ── READ: get_projects ───────────────────────────────────────────────────
    if name == "get_projects":
        status = args.get("status")
        try:
            query = supabase.table("projects").select("project_name, location, status")
            if status:
                query = query.ilike("status", f"%{status}%")
            res  = query.order("created_at", desc=False).execute()
            rows = res.data or []
            logger.info("[TOOL DEBUG] name=get_projects rows=%d", len(rows))
            if not rows:
                return json.dumps({"projects": [], "count": 0, "message": "No projects found."})
            return json.dumps({"projects": rows, "count": len(rows)})
        except Exception:
            logger.exception("Supabase error in get_projects")
            return json.dumps({"error": "Database error fetching projects."})

    # ── READ: get_project_summary ────────────────────────────────────────────
    elif name == "get_project_summary":
        p_name = args.get("project_name")
        try:
            project_id, p_real_name = _resolve_project(supabase, p_name)
        except ValueError as ve:
            logger.info("[TOOL DEBUG] name=get_project_summary rows=0 error=%s", ve)
            return json.dumps({"error": str(ve)})
        except Exception:
            logger.exception("Supabase error in get_project_summary lookup")
            return json.dumps({"error": "Database error fetching project."})

        try:
            logs     = supabase.table("daily_logs").select("*").eq("project_id", project_id).execute().data or []
            expenses = supabase.table("expenses").select("*").eq("project_id", project_id).execute().data or []
            equip    = supabase.table("equipment_logs").select("*").eq("project_id", project_id).execute().data or []
        except Exception:
            logger.exception("Supabase error fetching project sub-tables")
            return json.dumps({"error": "Database error fetching project details."})

        total_trenching = sum(l.get("trenching_meters", 0) or 0 for l in logs)
        total_pipe      = sum(l.get("pipe_laid_meters", 0) or 0 for l in logs)
        total_spent     = sum(e.get("amount", 0) or 0 for e in expenses)

        summary_data = {
            "project_name":           p_real_name,
            "total_trenching_meters": total_trenching,
            "total_pipe_laid_meters": total_pipe,
            "total_expenses_inr":     total_spent,
            "equipment_logs_count":   len(equip),
            "total_daily_logs_count": len(logs)
        }
        logger.info("[TOOL DEBUG] name=get_project_summary rows=%d", len(logs))
        return json.dumps(summary_data)

    # ── READ: get_daily_logs ─────────────────────────────────────────────────
    elif name == "get_daily_logs":
        p_name = args.get("project_name")
        limit  = args.get("limit", 5)
        try:
            limit = max(1, min(int(limit), 50))
        except (TypeError, ValueError):
            limit = 5

        query = supabase.table("daily_logs").select(
            "log_date, trenching_meters, pipe_laid_meters, backfilling_meters, raw_notes"
        )
        project_matched = None
        if p_name:
            try:
                project_id, project_matched = _resolve_project(supabase, p_name)
                query = query.eq("project_id", project_id)
            except ValueError as ve:
                logger.info("[TOOL DEBUG] name=get_daily_logs rows=0 error=%s", ve)
                return json.dumps({"error": str(ve)})
            except Exception:
                logger.exception("Supabase error looking up project for get_daily_logs")
                return json.dumps({"error": "Database error fetching project."})

        try:
            res  = query.order("log_date", desc=True).limit(limit).execute()
            rows = res.data or []
            logger.info("[TOOL DEBUG] name=get_daily_logs rows=%d", len(rows))
            if not rows:
                return json.dumps({
                    "project_name": project_matched or "All Projects",
                    "daily_logs": [],
                    "message": "No daily logs found."
                })
            return json.dumps({
                "project_name": project_matched or "All Projects",
                "daily_logs": rows
            })
        except Exception:
            logger.exception("Supabase error in get_daily_logs")
            return json.dumps({"error": "Database error fetching daily logs."})

    # ── READ: get_recent_expenses ────────────────────────────────────────────
    elif name == "get_recent_expenses":
        p_name = args.get("project_name")
        limit  = args.get("limit", 5)
        try:
            limit = max(1, min(int(limit), 50))
        except (TypeError, ValueError):
            limit = 5

        query = supabase.table("expenses").select(
            "category, amount, payment_mode, vendor_or_recipient, expense_date, notes"
        )
        project_matched = None
        if p_name:
            try:
                project_id, project_matched = _resolve_project(supabase, p_name)
                query = query.eq("project_id", project_id)
            except ValueError as ve:
                logger.info("[TOOL DEBUG] name=get_recent_expenses rows=0 error=%s", ve)
                return json.dumps({"error": str(ve)})
            except Exception:
                logger.exception("Supabase error looking up project for get_recent_expenses")
                return json.dumps({"error": "Database error fetching project."})

        try:
            res  = query.order("expense_date", desc=True).limit(limit).execute()
            rows = res.data or []
            logger.info("[TOOL DEBUG] name=get_recent_expenses rows=%d", len(rows))
            if not rows:
                return json.dumps({
                    "project_name": project_matched or "All Projects",
                    "expenses": [],
                    "message": "No expenses found."
                })
            return json.dumps({
                "project_name": project_matched or "All Projects",
                "expenses": rows
            })
        except Exception:
            logger.exception("Supabase error in get_recent_expenses")
            return json.dumps({"error": "Database error fetching expenses."})

    # ── READ: get_equipment_logs ─────────────────────────────────────────────
    elif name == "get_equipment_logs":
        p_name = args.get("project_name")
        limit  = args.get("limit", 5)
        try:
            limit = max(1, min(int(limit), 50))
        except (TypeError, ValueError):
            limit = 5

        query = supabase.table("equipment_logs").select(
            "log_date, equipment_name, hours_operated, fuel_consumed_liters, operator_name, notes"
        )
        project_matched = None
        if p_name:
            try:
                project_id, project_matched = _resolve_project(supabase, p_name)
                query = query.eq("project_id", project_id)
            except ValueError as ve:
                logger.info("[TOOL DEBUG] name=get_equipment_logs rows=0 error=%s", ve)
                return json.dumps({"error": str(ve)})
            except Exception:
                logger.exception("Supabase error looking up project for get_equipment_logs")
                return json.dumps({"error": "Database error fetching project."})

        try:
            res  = query.order("log_date", desc=True).limit(limit).execute()
            rows = res.data or []
            logger.info("[TOOL DEBUG] name=get_equipment_logs rows=%d", len(rows))
            if not rows:
                return json.dumps({
                    "project_name": project_matched or "All Projects",
                    "equipment_logs": [],
                    "message": "No equipment logs found."
                })
            return json.dumps({
                "project_name": project_matched or "All Projects",
                "equipment_logs": rows
            })
        except Exception:
            logger.exception("Supabase error in get_equipment_logs")
            return json.dumps({"error": "Database error fetching equipment logs."})

    # ── READ: get_project_attributes ─────────────────────────────────────────
    elif name == "get_project_attributes":
        p_name         = args.get("project_name")
        attribute_name = args.get("attribute_name")

        try:
            project_id, project_matched = _resolve_project(supabase, p_name)
        except ValueError as ve:
            logger.info("[TOOL DEBUG] name=get_project_attributes rows=0 error=%s", ve)
            return json.dumps({"error": str(ve)})
        except Exception:
            logger.exception("Supabase error in get_project_attributes project lookup")
            return json.dumps({"error": "Database error fetching project."})

        try:
            query = supabase.table("project_attributes").select(
                "attribute_name, attribute_type, attribute_value, created_at"
            ).eq("project_id", project_id)
            if attribute_name:
                query = query.ilike("attribute_name", attribute_name.strip())
            res  = query.order("created_at", desc=True).execute()
            rows = res.data or []
            logger.info("[TOOL DEBUG] name=get_project_attributes rows=%d", len(rows))
            if not rows:
                return json.dumps({
                    "project_name": project_matched,
                    "attributes": [],
                    "message": "No attributes found for this project."
                })
            return json.dumps({
                "project_name": project_matched,
                "attributes": rows
            })
        except Exception:
            logger.exception("Supabase error in get_project_attributes")
            return json.dumps({"error": "Database error fetching project attributes."})

    # ── WRITE: create_daily_log ──────────────────────────────────────────────
    elif name == "create_daily_log":
        p_name   = args.get("project_name")
        log_date = args.get("log_date")

        if not p_name:
            return json.dumps({"error": "project_name is required."})
        if not log_date:
            return json.dumps({"error": "log_date is required (YYYY-MM-DD)."})

        try:
            project_id, p_real_name = _resolve_project(supabase, p_name)
        except ValueError as ve:
            return json.dumps({"error": str(ve)})
        except Exception:
            logger.exception("Supabase error in create_daily_log project lookup")
            return json.dumps({"error": "Database error resolving project."})

        try:
            log_date = _validate_date(log_date, "log_date")
        except ValueError as ve:
            return json.dumps({"error": str(ve)})

        row = {"project_id": project_id, "log_date": log_date}
        for field in ("trenching_meters", "pipe_laid_meters", "backfilling_meters"):
            val = args.get(field)
            if val is not None:
                try:
                    row[field] = _validate_non_negative_number(val, field)
                except ValueError as ve:
                    return json.dumps({"error": str(ve)})
        raw_notes = args.get("raw_notes")
        if raw_notes:
            row["raw_notes"] = str(raw_notes).strip()

        try:
            res      = supabase.table("daily_logs").insert(row).execute()
            inserted = res.data[0] if res.data else row
            logger.info("[TOOL DEBUG] name=create_daily_log action=write rows=1 project=%s date=%s",
                        p_real_name, log_date)
            return json.dumps({
                "success": True,
                "message": f"Daily log created for '{p_real_name}' on {log_date}.",
                "record":  inserted
            })
        except Exception:
            logger.exception("Supabase error inserting daily log")
            return json.dumps({"error": "Database error creating daily log."})

    # ── WRITE: create_expense ────────────────────────────────────────────────
    elif name == "create_expense":
        p_name       = args.get("project_name")
        expense_date = args.get("expense_date")
        category     = args.get("category")
        amount       = args.get("amount")
        payment_mode = args.get("payment_mode")

        missing = [f for f, v in [
            ("project_name", p_name), ("expense_date", expense_date),
            ("category", category), ("amount", amount), ("payment_mode", payment_mode)
        ] if not v and v != 0]
        if missing:
            return json.dumps({"error": f"Missing required fields: {', '.join(missing)}."})

        try:
            project_id, p_real_name = _resolve_project(supabase, p_name)
        except ValueError as ve:
            return json.dumps({"error": str(ve)})
        except Exception:
            logger.exception("Supabase error in create_expense project lookup")
            return json.dumps({"error": "Database error resolving project."})

        try:
            expense_date = _validate_date(expense_date, "expense_date")
        except ValueError as ve:
            return json.dumps({"error": str(ve)})

        try:
            amount = _validate_non_negative_number(amount, "amount")
        except ValueError as ve:
            return json.dumps({"error": str(ve)})

        row = {
            "project_id":   project_id,
            "expense_date": expense_date,
            "category":     str(category).strip(),
            "amount":       amount,
            "payment_mode": str(payment_mode).strip()
        }
        vendor = args.get("vendor_or_recipient")
        if vendor:
            row["vendor_or_recipient"] = str(vendor).strip()
        notes = args.get("notes")
        if notes:
            row["notes"] = str(notes).strip()

        try:
            res      = supabase.table("expenses").insert(row).execute()
            inserted = res.data[0] if res.data else row
            logger.info("[TOOL DEBUG] name=create_expense action=write rows=1 project=%s date=%s amount=%s",
                        p_real_name, expense_date, amount)
            return json.dumps({
                "success": True,
                "message": f"Expense of {amount} INR ({category}) recorded for '{p_real_name}' on {expense_date}.",
                "record":  inserted
            })
        except Exception:
            logger.exception("Supabase error inserting expense")
            return json.dumps({"error": "Database error creating expense."})

    # ── WRITE: create_equipment_log ──────────────────────────────────────────
    elif name == "create_equipment_log":
        p_name         = args.get("project_name")
        log_date       = args.get("log_date")
        equipment_name = args.get("equipment_name")

        if not p_name:
            return json.dumps({"error": "project_name is required."})
        if not log_date:
            return json.dumps({"error": "log_date is required (YYYY-MM-DD)."})
        if not equipment_name:
            return json.dumps({"error": "equipment_name is required."})

        try:
            project_id, p_real_name = _resolve_project(supabase, p_name)
        except ValueError as ve:
            return json.dumps({"error": str(ve)})
        except Exception:
            logger.exception("Supabase error in create_equipment_log project lookup")
            return json.dumps({"error": "Database error resolving project."})

        try:
            log_date = _validate_date(log_date, "log_date")
        except ValueError as ve:
            return json.dumps({"error": str(ve)})

        row = {
            "project_id":     project_id,
            "log_date":       log_date,
            "equipment_name": str(equipment_name).strip()
        }
        for field in ("hours_operated", "fuel_consumed_liters"):
            val = args.get(field)
            if val is not None:
                try:
                    row[field] = _validate_non_negative_number(val, field)
                except ValueError as ve:
                    return json.dumps({"error": str(ve)})
        operator = args.get("operator_name")
        if operator:
            row["operator_name"] = str(operator).strip()
        notes = args.get("notes")
        if notes:
            row["notes"] = str(notes).strip()

        try:
            res      = supabase.table("equipment_logs").insert(row).execute()
            inserted = res.data[0] if res.data else row
            logger.info("[TOOL DEBUG] name=create_equipment_log action=write rows=1 project=%s date=%s equip=%s",
                        p_real_name, log_date, equipment_name)
            return json.dumps({
                "success": True,
                "message": f"Equipment log for '{equipment_name}' created for '{p_real_name}' on {log_date}.",
                "record":  inserted
            })
        except Exception:
            logger.exception("Supabase error inserting equipment log")
            return json.dumps({"error": "Database error creating equipment log."})

    # ── WRITE: set_project_attribute ─────────────────────────────────────────
    # Schema: project_attributes(id UUID, project_id UUID REFERENCES projects(id),
    #         attribute_name TEXT, attribute_type TEXT, attribute_value TEXT, created_at TIMESTAMP)
    # Attributes belong to PROJECTS, not to DB columns.
    # No duplicate attributes per project — update if exists, insert if missing.
    elif name == "set_project_attribute":
        p_name          = args.get("project_name")
        attribute_name  = args.get("attribute_name")
        attribute_value = args.get("attribute_value")
        attribute_type  = args.get("attribute_type", "text")

        logger.info("[WRITE DEBUG] tool=%s", name)
        logger.info("[WRITE DEBUG] project=%s", p_name)
        logger.info("[WRITE DEBUG] attribute=%s", attribute_name)
        logger.info("[WRITE DEBUG] value=%s", attribute_value)

        if not p_name:
            return json.dumps({"error": "project_name is required."})
        if not attribute_name or not attribute_name.strip():
            return json.dumps({"error": "attribute_name is required."})
        if attribute_value is None:
            return json.dumps({"error": "attribute_value is required."})

        # Sanitise attribute_type
        if attribute_type not in ("text", "numeric", "date"):
            attribute_type = "text"

        # Normalise attribute_name to safe snake_case
        safe_attr_name = re.sub(r"[^\w]", "_", attribute_name.strip().lower())
        attribute_value_str = str(attribute_value).strip()

        # Validate value matches declared type
        if attribute_type == "numeric":
            try:
                float(attribute_value_str)
            except ValueError:
                return json.dumps({
                    "error": (
                        f"attribute_value '{attribute_value_str}' is not a valid number "
                        f"for attribute_type='numeric'."
                    )
                })
        elif attribute_type == "date":
            try:
                _validate_date(attribute_value_str, "attribute_value")
            except ValueError as ve:
                return json.dumps({"error": str(ve)})

        # Resolve project
        try:
            project_id, p_real_name = _resolve_project(supabase, p_name)
            logger.info("[WRITE DEBUG] project_id=%s", project_id)
        except ValueError as ve:
            return json.dumps({"error": str(ve)})
        except Exception:
            logger.exception("Supabase error in set_project_attribute project lookup")
            return json.dumps({"error": "Database error resolving project."})

        # Check for existing attribute (prevents duplicates)
        try:
            existing_res = (
                supabase.table("project_attributes")
                .select("id, attribute_value")
                .eq("project_id", project_id)
                .eq("attribute_name", safe_attr_name)
                .execute()
            )
            existing_rows = existing_res.data or []
        except Exception:
            logger.exception("Supabase error checking existing project_attributes")
            return json.dumps({"error": "Database error checking existing attributes."})

        try:
            if existing_rows:
                # UPDATE — no duplicate created
                existing_id = existing_rows[0]["id"]
                old_value   = existing_rows[0].get("attribute_value", "")
                write_res   = (
                    supabase.table("project_attributes")
                    .update({
                        "attribute_value": attribute_value_str,
                        "attribute_type":  attribute_type
                    })
                    .eq("id", existing_id)
                    .execute()
                )
                action = "updated"
                detail = f"(was '{old_value}', now '{attribute_value_str}')"
            else:
                # INSERT — new attribute for this project
                write_res  = (
                    supabase.table("project_attributes")
                    .insert({
                        "project_id":      project_id,
                        "attribute_name":  safe_attr_name,
                        "attribute_type":  attribute_type,
                        "attribute_value": attribute_value_str
                    })
                    .execute()
                )
                action = "created"
                detail = f"(value='{attribute_value_str}')"

            if not write_res.data:
                logger.error(
                    "[WRITE DEBUG] insert_result=error empty_data project=%s attr=%s",
                    p_real_name, safe_attr_name
                )
                return json.dumps({
                    "error": (
                        "Write returned no data - insert/update did not persist "
                        "(possible RLS or constraint issue)."
                    )
                })

            logger.info("[WRITE DEBUG] insert_result=success")
            updated_row = write_res.data[0]

            # Verify by reading back the row we just wrote
            verify_res = (
                supabase.table("project_attributes")
                .select("id")
                .eq("id", updated_row.get("id"))
                .execute()
            )
            verified = bool(verify_res.data)
            logger.info("[WRITE DEBUG] verified_in_db=%s", verified)

            if not verified:
                return json.dumps({
                    "error": (
                        "Write reported success but row not found on read-back "
                        "(transaction aborted or rolled back)."
                    )
                })

            logger.info("[WRITE DEBUG] supabase_write=true")
            return json.dumps({
                "success": True,
                "message": (
                    f"Attribute '{safe_attr_name}' {action} for project "
                    f"'{p_real_name}' {detail}."
                ),
                "record": updated_row
            })
        except Exception:
            logger.info("[WRITE DEBUG] insert_result=error exception")
            logger.info("[WRITE DEBUG] supabase_write=false")
            logger.exception("Supabase error upserting project_attributes")
            return json.dumps({"error": "Database error saving project attribute."})

    # ── Unknown tool ─────────────────────────────────────────────────────────
    logger.warning("[TOOL DEBUG] name=%s — unknown tool called", name)
    return json.dumps({"error": f"Unknown tool: '{name}'."})


# ---------------------------------------------------------------------------
# Meta WhatsApp Cloud API — send message
# ---------------------------------------------------------------------------
def send_whatsapp_message(to, message):
    """Send a text message via Meta WhatsApp Cloud API (Graph API)."""
    if not META_ACCESS_TOKEN or not META_PHONE_NUMBER_ID:
        raise RuntimeError("META_ACCESS_TOKEN or META_PHONE_NUMBER_ID env var is missing")

    url = f"https://graph.facebook.com/{META_GRAPH_API_VERSION}/{META_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {META_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": message}
    }

    masked_recipient = str(to)[:4] + "***" if to else "unknown"
    masked_phone_id  = str(META_PHONE_NUMBER_ID)[:4] + "***" if META_PHONE_NUMBER_ID else "unknown"

    logger.info("[META] Sending response to: %s", masked_recipient)
    logger.info("[META] phone_number_id=%s message_type=text", masked_phone_id)
    logger.info("[PIPELINE] Meta HTTP POST to graph.facebook.com (timeout=30s)")

    response = requests.post(url, headers=headers, json=payload, timeout=30)
    logger.info("[META DEBUG] status=%s", response.status_code)
    logger.info("[META DEBUG] response_body=%s", response.text[:1000])

    if not response.ok:
        logger.error(
            "[PIPELINE] Meta send-message HTTP error: status=%s body=%s",
            response.status_code,
            response.text[:500]   # truncated — never contains the token
        )
        response.raise_for_status()

    try:
        resp_json     = response.json()
        messages_resp = resp_json.get("messages", [])
        if messages_resp and "id" in messages_resp[0]:
            logger.info("[META] WhatsApp message ID: %s", messages_resp[0]["id"])
        else:
            logger.info("[META] Message accepted by Meta, response: %s", response.text)
        return resp_json
    except Exception:
        logger.info("[META] Response body: %s", response.text)
        return {}


# ---------------------------------------------------------------------------
# AI agent processing
# ---------------------------------------------------------------------------
def process_whatsapp_message(sender_id, incoming_msg, message_id):
    """Run the incoming message through the Groq AI agent and reply via Meta."""
    print("[PIPELINE] -- START process_whatsapp_message --", flush=True)
    logger.info("[PIPELINE] -- START process_whatsapp_message --")
    logger.info("[WRITE DEBUG] incoming_text=%s", incoming_msg)
    logger.info("[PIPELINE] ── START id=%s from=%s*** ──", message_id, str(sender_id)[:4])

    # Diagnostic: log model and current message (no private data)
    logger.info("[LLM DEBUG] model=%s", GROQ_MODEL)
    logger.info("[LLM DEBUG] current_message=%s", incoming_msg)
    logger.info("[PIPELINE] Extracted message text (len=%d)", len(incoming_msg))
    logger.info("[PIPELINE] Sender ID prefix: %s***", str(sender_id)[:4])

    # ── Stage 1: Supabase client init ────────────────────────────────────────
    print("[PIPELINE] Stage 1 starting", flush=True)
    logger.info("[PIPELINE] Stage 1 starting — Initialising Supabase client")
    try:
        supabase = get_supabase()
        print("[PIPELINE] Stage 1 completed", flush=True)
        logger.info("[PIPELINE] Stage 1 completed — Supabase client OK")
    except Exception:
        print("[PIPELINE] Stage 1 FAILED", flush=True)
        traceback.print_exc()
        logger.exception("[PIPELINE] Stage 1 FAILED — could not create Supabase client")
        _send_error_reply(sender_id)
        return False

    # ── Stage 2: CRM conversation history ────────────────────────────────────
    print("[PIPELINE] Stage 2 starting", flush=True)
    logger.info("[PIPELINE] Stage 2 starting — Starting CRM history lookup")
    try:
        history_res = (
            supabase.table("chat_sessions")
            .select("role, message_text")
            .eq("whatsapp_number", sender_id)
            .order("created_at", desc=True)
            .limit(6)
            .execute()
        )
        print("[PIPELINE] Stage 2 completed", flush=True)
        raw_history = history_res.data or []
        logger.info("[PIPELINE] Stage 2 completed — CRM lookup completed, rows=%d", len(raw_history))
    except Exception:
        print("[PIPELINE] Stage 2 FAILED", flush=True)
        traceback.print_exc()
        logger.exception("[PIPELINE] Stage 2 FAILED — Supabase history fetch error")
        raw_history = []

    # Build clean alternating history (chronological, skip error/empty rows)
    chronological_history = list(reversed(raw_history))
    cleaned_history = []
    for h in chronological_history:
        role    = h.get("role")
        content = h.get("message_text", "").strip()
        if role in ("user", "assistant") and content:
            if (cleaned_history
                    and cleaned_history[-1]["role"] == role
                    and cleaned_history[-1]["content"] == content):
                continue
            cleaned_history.append({"role": role, "content": content})

    # Drop trailing user turn — the current message is appended fresh below
    if cleaned_history and cleaned_history[-1]["role"] == "user":
        cleaned_history.pop()

    last_role        = cleaned_history[-1]["role"] if cleaned_history else "none"
    last_msg_snippet = cleaned_history[-1]["content"][:60] if cleaned_history else "none"

    logger.info("[LLM DEBUG] history_count=%d", len(cleaned_history))
    logger.info("[LLM DEBUG] history_last_role=%s", last_role)
    logger.info("[LLM DEBUG] history_last_message=%s", last_msg_snippet)

    # ── Build message list for LLM ────────────────────────────────────────────
    # DB Grounding rules (Requirement #3): fresh Supabase data only, no caching,
    # no hallucination. Greetings MUST NOT trigger DB tools.
    system_prompt = (
        "You are an intelligent construction field ops assistant. "
        "Answer concisely for WhatsApp (short, clear, bold where helpful).\n\n"
        "CRITICAL GROUNDING RULES:\n"
        "1. For greetings, pleasantries, or general openers (e.g. 'Hello', 'Hi', 'Hey', "
        "'Help', 'Who are you?') respond warmly WITHOUT calling any database tool.\n"
        "2. For ANY query about projects, daily logs, expenses, equipment, or project "
        "attributes — ALWAYS call the appropriate database tool first. "
        "NEVER answer from memory or prior context.\n"
        "3. Your final answer MUST be grounded EXCLUSIVELY in data returned by the "
        "tool calls in the current conversation turn. "
        "NEVER invent, assume, cache, reuse, or guess CRM data.\n"
        "4. If a tool returns 0 rows or an error message, clearly state "
        "'No records found' or relay the exact error. Never fabricate data.\n"
        "5. For write operations (create log, expense, equipment log, set attribute): "
        "extract structured arguments from the user message, call the appropriate "
        "tool, then confirm success or report the error returned by the tool.\n"
        "6. NEVER expose <think> reasoning, raw tool JSON, or internal arguments to the user."
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(cleaned_history)
    messages.append({"role": "user", "content": incoming_msg})
    logger.info("[PIPELINE] LLM context built: %d message(s) (incl. system)", len(messages))

    # ── Stage 3: Persist user message ────────────────────────────────────────
    print("[PIPELINE] Stage 3 starting", flush=True)
    logger.info("[PIPELINE] Stage 3 starting — Persisting user message to Supabase")
    try:
        supabase.table("chat_sessions").insert({
            "whatsapp_number": sender_id,
            "role":            "user",
            "message_text":    incoming_msg
        }).execute()
        print("[PIPELINE] Stage 3 completed", flush=True)
        logger.info("[PIPELINE] Stage 3 completed — User message persisted OK")
    except Exception:
        print("[PIPELINE] Stage 3 FAILED", flush=True)
        traceback.print_exc()
        logger.exception("[PIPELINE] Stage 3 FAILED — Supabase insert user message error")
        # Non-fatal: continue even if history logging fails

    # ── Stage 4: Groq client init ─────────────────────────────────────────────
    print("[PIPELINE] Stage 4 starting", flush=True)
    logger.info("[PIPELINE] Stage 4 starting — Initialising Groq client")
    try:
        groq_client = get_groq()
        print("[PIPELINE] Stage 4 completed", flush=True)
        logger.info("[PIPELINE] Stage 4 completed — Groq client OK")
    except Exception:
        print("[PIPELINE] Stage 4 FAILED", flush=True)
        traceback.print_exc()
        logger.exception("[PIPELINE] Stage 4 FAILED — could not create Groq client")
        _send_error_reply(sender_id)
        return False

    # ── Stage 5: First LLM call ───────────────────────────────────────────────
    print("[PIPELINE] Stage 5 starting", flush=True)
    logger.info("[PIPELINE] Stage 5 starting — Starting LLM call (model=%s, messages=%d)",
                GROQ_MODEL, len(messages))
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            timeout=25
        )
        print("[PIPELINE] Stage 5 completed", flush=True)
        logger.info("[WRITE DEBUG] llm_called=true")
        logger.info("[PIPELINE] Stage 5 completed — LLM call completed")
    except Exception:
        print("[PIPELINE] Stage 5 FAILED", flush=True)
        logger.info("[WRITE DEBUG] llm_called=false")
        traceback.print_exc()
        logger.exception("[PIPELINE] Stage 5 FAILED — Groq first LLM call error")
        _send_error_reply(sender_id)
        return False

    msg_obj        = response.choices[0].message
    has_tool_calls = bool(msg_obj.tool_calls)
    logger.info("[WRITE DEBUG] tool_called=%s", str(has_tool_calls).lower())
    logger.info("[TOOL DEBUG] called=%s", str(has_tool_calls).lower())
    logger.info("[PIPELINE] Stage 5 — finish_reason=%s tool_calls=%s",
                response.choices[0].finish_reason, has_tool_calls)

    # ── Stage 6: Tool execution (if requested by LLM) ────────────────────────
    if has_tool_calls:
        print("[PIPELINE] Stage 6 starting", flush=True)
        logger.info("[PIPELINE] Stage 6 starting")
        tool_names = [tc.function.name for tc in msg_obj.tool_calls]
        logger.info("[PIPELINE] Stage 6 — Tool calls requested: %s", tool_names)

        messages.append({
            "role":       "assistant",
            "content":    msg_obj.content,
            "tool_calls": [
                {
                    "id":   tc.id,
                    "type": "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in msg_obj.tool_calls
            ]
        })

        for tool_call in msg_obj.tool_calls:
            tool_name = tool_call.function.name
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                logger.error("[PIPELINE] Stage 6 — JSON decode error on tool args for %s", tool_name)
                tool_args = {}

            logger.info("[TOOL DEBUG] name=%s", tool_name)
            logger.info("[TOOL DEBUG] arguments_keys=%s", list(tool_args.keys()))
            logger.info("[PIPELINE] Stage 6 — Executing tool: %s", tool_name)

            try:
                tool_output = execute_tool(tool_name, tool_args)
                logger.info("[PIPELINE] Stage 6 — Tool '%s' completed, output_len=%d",
                            tool_name, len(tool_output))
            except Exception:
                print(f"[PIPELINE] Stage 6 tool {tool_name} FAILED", flush=True)
                traceback.print_exc()
                logger.exception("[PIPELINE] Stage 6 FAILED — tool %s raised exception", tool_name)
                tool_output = json.dumps({"error": "Tool execution failed."})

            # Pass tool result back to LLM — required before final answer (Requirement #4)
            messages.append({
                "role":         "tool",
                "tool_call_id": tool_call.id,
                "content":      tool_output
            })

        print("[PIPELINE] Stage 6 completed", flush=True)
        logger.info("[PIPELINE] Stage 6 completed")

        # ── Stage 7: Second LLM call (final answer grounded in tool results) ─
        print("[PIPELINE] Stage 7 starting", flush=True)
        logger.info("[PIPELINE] Stage 7 starting — Starting second LLM call (after tools)")
        try:
            second_response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                timeout=25
            )
            raw_reply  = second_response.choices[0].message.content or ""
            logger.info(
                "[LLM DEBUG] Stage 7 raw_reply (first 300 chars)=%s",
                raw_reply[:300]
            )
            # Strip any <think>...</think> blocks before sending (Requirement #2)
            reply_text = sanitize_reply(raw_reply)
            print("[PIPELINE] Stage 7 completed", flush=True)
            logger.info("[PIPELINE] Stage 7 completed — final reply len=%d (raw len=%d)",
                        len(reply_text), len(raw_reply))
        except Exception:
            print("[PIPELINE] Stage 7 FAILED", flush=True)
            traceback.print_exc()
            logger.exception("[PIPELINE] Stage 7 FAILED — Groq second LLM call error")
            _send_error_reply(sender_id)
            return False
    else:
        print("[PIPELINE] Stage 6 starting", flush=True)
        print("[PIPELINE] Stage 6 completed", flush=True)
        logger.info("[PIPELINE] Stage 6 — No tool calls, using direct LLM answer")
        raw_reply  = msg_obj.content or ""
        # Strip any <think>...</think> blocks even for direct non-tool answers (Requirement #2)
        reply_text = sanitize_reply(raw_reply)
        logger.info("[PIPELINE] Direct reply len=%d (raw len=%d)", len(reply_text), len(raw_reply))

    print("[PIPELINE] Generated response", flush=True)
    logger.info("[PIPELINE] Generated response (len=%d chars)", len(reply_text or ""))

    # ── Stage 8: Persist sanitized assistant reply ───────────────────────────
    print("[PIPELINE] Stage 8 starting", flush=True)
    logger.info("[PIPELINE] Stage 8 starting — Persisting assistant reply to Supabase")
    try:
        supabase.table("chat_sessions").insert({
            "whatsapp_number": sender_id,
            "role":            "assistant",
            "message_text":    reply_text   # always the sanitized version — no <think> content
        }).execute()
        print("[PIPELINE] Stage 8 completed", flush=True)
        logger.info("[PIPELINE] Stage 8 completed — Assistant reply persisted OK")
    except Exception:
        print("[PIPELINE] Stage 8 FAILED", flush=True)
        traceback.print_exc()
        logger.exception("[PIPELINE] Stage 8 FAILED — Supabase insert assistant message error")
        # Non-fatal: still attempt to send the reply

    # ── Stage 8.5: Cap reply to WhatsApp's 4096-character limit ───────────────
    logger.info("[WHATSAPP] reply_length=%d", len(reply_text))
    capped_text, was_truncated = cap_whatsapp_reply(reply_text)
    logger.info("[WHATSAPP] truncated=%s", str(was_truncated).lower())
    reply_text = capped_text

    # ── Stage 9: Send reply via Meta ──────────────────────────────────────────
    print("[PIPELINE] Stage 9 starting", flush=True)
    logger.info("[PIPELINE] Stage 9 starting — Starting Meta send")
    if not reply_text:
        logger.warning(
            "[PIPELINE] Stage 9 — reply_text empty; sending placeholder to avoid Meta 400"
        )
        reply_text = "Sorry, no response generated."
    try:
        send_whatsapp_message(sender_id, reply_text)
        print("[PIPELINE] Stage 9 completed", flush=True)
        logger.info("[PIPELINE] Stage 9 completed")
    except Exception:
        print("[PIPELINE] Stage 9 FAILED", flush=True)
        traceback.print_exc()
        logger.exception("[PIPELINE] Stage 9 FAILED — Meta send-message error")
        return False

    logger.info("[PIPELINE] ── END id=%s SUCCESS ──", message_id)
    return True


def _send_error_reply(sender_id):
    """Attempt to send a generic error message back to the user via Meta."""
    try:
        send_whatsapp_message(
            sender_id,
            "Sorry, I encountered an error processing your request. Please try again."
        )
    except Exception:
        logger.exception("[PIPELINE] Meta send-message error while sending error reply")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def health():
    """Simple health-check endpoint."""
    return {"status": "ok", "service": "Construction AI Agent"}, 200


@app.route("/webhook/whatsapp", methods=["GET"], strict_slashes=False)
def verify_whatsapp_webhook():
    """
    Meta webhook verification handshake.
    Meta sends: hub.mode, hub.verify_token, hub.challenge
    """
    mode      = request.args.get("hub.mode")
    token     = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if not mode and not token:
        # Plain browser/health probe — no verification params
        return "WhatsApp Webhook Endpoint is Active!", 200

    if mode == "subscribe" and token == META_VERIFY_TOKEN:
        logger.info("[WEBHOOK] Meta verification successful")
        return challenge, 200

    logger.warning("[WEBHOOK] Meta verification FAILED: mode=%s token_match=%s",
                   mode, token == META_VERIFY_TOKEN)
    return "Forbidden", 403


@app.route("/webhook/whatsapp", methods=["POST"], strict_slashes=False)
def receive_whatsapp_webhook():
    """
    Receives incoming WhatsApp messages and status events from Meta.
    """
    print("=== META WHATSAPP WEBHOOK POST RECEIVED ===", flush=True)
    logger.info("=== META WHATSAPP WEBHOOK POST RECEIVED ===")
    logger.info("[WRITE DEBUG] webhook_reached=true")
    logger.info("[WEBHOOK] POST received")
    logger.info("[WEBHOOK] method=%s path=%s content_type=%s",
                request.method, request.path, request.content_type)

    try:
        data = request.get_json(silent=True) or {}
    except Exception:
        logger.exception("[WEBHOOK] Failed to parse JSON body")
        return "OK", 200  # always return 200 to Meta

    entries = data.get("entry", [])
    logger.info("[WEBHOOK] object=%s", data.get("object"))
    logger.info("[WEBHOOK] entry_count=%d", len(entries))

    try:
        for entry in entries:
            for change in entry.get("changes", []):
                field    = change.get("field", "unknown")
                value    = change.get("value", {})
                statuses = value.get("statuses", [])
                msgs     = value.get("messages", [])

                logger.info("[WEBHOOK] field=%s", field)
                logger.info("[WEBHOOK] message_event=%s", str(bool(msgs)).lower())

                # Status/delivery/read events — acknowledge and skip
                if statuses:
                    for status in statuses:
                        logger.info(
                            "[WEBHOOK] Status event: id=%s status=%s",
                            status.get("id"), status.get("status")
                        )
                    continue

                if not msgs:
                    logger.info("[WEBHOOK] No messages in this change entry, skipping")
                    continue

                msg        = msgs[0]
                message_id = msg.get("id", "unknown")
                sender_id  = msg.get("from", "")
                msg_type   = msg.get("type", "unknown")

                masked_sender = str(sender_id)[:4] + "***" if sender_id else "unknown"

                logger.info("[WEBHOOK] REAL MESSAGE RECEIVED")
                logger.info("[WEBHOOK] message_id=%s", message_id)
                logger.info("[WEBHOOK] sender=%s", masked_sender)
                logger.info("[WEBHOOK] type=%s", msg_type)

                if msg_type != "text":
                    logger.info(
                        "[WEBHOOK] Non-text type '%s' from %s — acknowledged, not processed",
                        msg_type, masked_sender
                    )
                    continue

                incoming_msg = msg.get("text", {}).get("body", "").strip()
                if not incoming_msg:
                    logger.warning("[WEBHOOK] Empty text body from %s — skipping", masked_sender)
                    continue

                logger.info("[WEBHOOK] Message received, handing to pipeline")
                print("[PIPELINE] ABOUT TO CALL process_whatsapp_message()", flush=True)
                logger.info("[PIPELINE] ABOUT TO CALL process_whatsapp_message()")
                try:
                    process_whatsapp_message(sender_id, incoming_msg, message_id)
                    print("[PIPELINE] process_whatsapp_message() RETURNED", flush=True)
                    logger.info("[PIPELINE] process_whatsapp_message() RETURNED")
                except Exception:
                    print("[PIPELINE] EXCEPTION while calling process_whatsapp_message()", flush=True)
                    logger.exception("[PIPELINE] EXCEPTION while calling process_whatsapp_message()")
                    traceback.print_exc()

    except Exception:
        logger.exception("[WEBHOOK] Unexpected error in webhook processing")

    return "OK", 200


@app.route("/debug/chat_sessions", methods=["GET"])
def debug_chat_sessions():
    try:
        supabase = get_supabase()
        res = supabase.table("chat_sessions").select("*").order("created_at", desc=True).limit(10).execute()
        return {"count": len(res.data or []), "recent": res.data}
    except Exception as e:
        logger.exception("Debug chat_sessions failed")
        return {"error": str(e)}, 500