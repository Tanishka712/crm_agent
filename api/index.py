import os
import json
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

GROQ_MODEL = "llama-3.3-70b-versatile"

# Log safe masked configuration
logger.info(
    "[META CONFIG] phone_number_id=%s*** waba_id=%s*** webhook_path=/webhook/whatsapp",
    str(META_PHONE_NUMBER_ID)[:4] if META_PHONE_NUMBER_ID else "none",
    str(META_WHATSAPP_BUSINESS_ACCOUNT_ID)[:4] if META_WHATSAPP_BUSINESS_ACCOUNT_ID else "none"
)

# ---------------------------------------------------------------------------
# Lazy client initialisation
# Clients are created on first use, not at module load time.
# This prevents a startup crash from killing ALL routes (including POST)
# when an env var is missing or a dependency fails to initialise.
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
# Tool schemas
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_projects",
            "description": "Fetch the list of construction projects from the database (including project name, location, and status). Use this whenever the user asks about active projects, ongoing projects, project list, or what projects exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Optional status filter (e.g. 'Active'). If omitted, returns all projects."
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
            "description": "Fetch overall progress totals, total expenses, and equipment stats for a given project name.",
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
            "description": "Get daily site progress logs (including trenching meters, pipe laid meters, backfilling meters, and raw notes) for a project or all projects, ordered by date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_name": {
                        "type": "string",
                        "description": "Name of the project to fetch daily logs for (e.g. 'Pipeline Alpha')."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of daily log entries to return (default 5, max 50).",
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
            "description": "Get site expenses (category, amount, payment mode, vendor/recipient, date, notes). Can optionally filter by project name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_name": {
                        "type": "string",
                        "description": "Optional project name to filter expenses for (e.g. 'Pipeline Alpha')."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of expenses to return, e.g. 5 or 10.",
                        "minimum": 1,
                        "maximum": 50
                    }
                },
                "required": []
            }
        }
    }
]

# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------
def execute_tool(name, args):
    supabase = get_supabase()

    if name == "get_projects":
        status = args.get("status")
        try:
            query = supabase.table("projects").select("project_name, location, status")
            if status:
                query = query.ilike("status", f"%{status}%")
            res = query.order("created_at", desc=False).execute()
            rows = res.data or []
            logger.info(
                "[DB TEST] tool=get_projects status=%s rows=%d rows_data=%s result_passed_to_llm=true",
                status or "All", len(rows), rows
            )
            return json.dumps({
                "projects": rows,
                "count": len(rows)
            })
        except Exception:
            logger.exception("Supabase error in get_projects")
            return json.dumps({"error": "Database error fetching projects."})

    elif name == "get_project_summary":
        p_name = args.get("project_name")
        try:
            res = supabase.table("projects").select("*").ilike("project_name", f"%{p_name}%").execute()
        except Exception:
            logger.exception("Supabase error in get_project_summary")
            return json.dumps({"error": "Database error fetching project."})

        if not res.data:
            logger.info("[DB TEST] tool=get_project_summary project=%s rows=0 rows_data=[] result_passed_to_llm=true", p_name)
            return json.dumps({"error": f"No project found matching '{p_name}'"})

        project_id = res.data[0]["id"]
        p_real_name = res.data[0]["project_name"]

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
            "project_name": p_real_name,
            "total_trenching_meters": total_trenching,
            "total_pipe_laid_meters": total_pipe,
            "total_expenses_inr": total_spent,
            "equipment_logs_count": len(equip),
            "total_daily_logs_count": len(logs)
        }
        logger.info(
            "[DB TEST] tool=get_project_summary project=%s rows=%d rows_data=%s result_passed_to_llm=true",
            p_real_name, len(logs), summary_data
        )
        return json.dumps(summary_data)

    elif name == "get_daily_logs":
        p_name = args.get("project_name")
        limit = args.get("limit", 5)
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 5
        limit = max(1, min(limit, 50))

        query = supabase.table("daily_logs").select(
            "log_date, trenching_meters, pipe_laid_meters, backfilling_meters, raw_notes"
        )
        project_matched = None
        if p_name:
            try:
                p_res = supabase.table("projects").select("id, project_name").ilike("project_name", f"%{p_name}%").execute()
                if not p_res.data:
                    logger.info("[DB TEST] tool=get_daily_logs project=%s rows=0 rows_data=[] result_passed_to_llm=true", p_name)
                    return json.dumps({"error": f"No project found matching '{p_name}'"})
                pid = p_res.data[0]["id"]
                project_matched = p_res.data[0]["project_name"]
                query = query.eq("project_id", pid)
            except Exception:
                logger.exception("Supabase error looking up project for get_daily_logs")
                return json.dumps({"error": "Database error fetching project."})

        try:
            res = query.order("log_date", desc=True).limit(limit).execute()
            rows = res.data or []
            logger.info(
                "[DB TEST] tool=get_daily_logs project=%s rows=%d rows_data=%s result_passed_to_llm=true",
                project_matched or p_name or "All", len(rows), rows
            )
            return json.dumps({
                "project_name": project_matched or "All Projects",
                "daily_logs": rows
            })
        except Exception:
            logger.exception("Supabase error in get_daily_logs")
            return json.dumps({"error": "Database error fetching daily logs."})

    elif name == "get_recent_expenses":
        p_name = args.get("project_name")
        limit = args.get("limit", 5)
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 5
        limit = max(1, min(limit, 50))

        query = supabase.table("expenses").select(
            "category, amount, payment_mode, vendor_or_recipient, expense_date, notes"
        )
        project_matched = None
        if p_name:
            try:
                p_res = supabase.table("projects").select("id, project_name").ilike("project_name", f"%{p_name}%").execute()
                if not p_res.data:
                    logger.info("[DB TEST] tool=get_recent_expenses project=%s rows=0 rows_data=[] result_passed_to_llm=true", p_name)
                    return json.dumps({"error": f"No project found matching '{p_name}'"})
                pid = p_res.data[0]["id"]
                project_matched = p_res.data[0]["project_name"]
                query = query.eq("project_id", pid)
            except Exception:
                logger.exception("Supabase error looking up project for get_recent_expenses")
                return json.dumps({"error": "Database error fetching project."})

        try:
            res = query.order("expense_date", desc=True).limit(limit).execute()
            rows = res.data or []
            logger.info(
                "[DB TEST] tool=get_recent_expenses project=%s rows=%d rows_data=%s result_passed_to_llm=true",
                project_matched or p_name or "All", len(rows), rows
            )
            return json.dumps({
                "project_name": project_matched or "All Projects",
                "expenses": rows
            })
        except Exception:
            logger.exception("Supabase error in get_recent_expenses")
            return json.dumps({"error": "Database error fetching expenses."})

    return json.dumps({"error": "Unknown tool function."})

# ---------------------------------------------------------------------------
# Meta WhatsApp Cloud API — send message
# ---------------------------------------------------------------------------
def send_whatsapp_message(to, message):
    """Send a text message via Meta WhatsApp Cloud API (Graph API v21.0)."""
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
    masked_phone_id = str(META_PHONE_NUMBER_ID)[:4] + "***" if META_PHONE_NUMBER_ID else "unknown"

    logger.info("[META] Sending response to: %s", masked_recipient)
    logger.info("[META] Sending WhatsApp text response")
    logger.info("[META] phone_number_id = %s", masked_phone_id)
    logger.info("[META] recipient = %s", masked_recipient)
    logger.info("[META] message_type = text")

    logger.info("[PIPELINE] Meta HTTP POST to graph.facebook.com (timeout=30s)")
    response = requests.post(url, headers=headers, json=payload, timeout=30)

    print("[PIPELINE] Meta response body:", response.text)
    logger.info("[PIPELINE] Meta send-message response: HTTP %s", response.status_code)

    if not response.ok:
        logger.error(
            "[PIPELINE] Meta send-message HTTP error: status=%s body=%s",
            response.status_code,
            response.text[:500]   # truncated — never contains the token
        )
        response.raise_for_status()

    try:
        resp_json = response.json()
        messages_resp = resp_json.get("messages", [])
        if messages_resp and "id" in messages_resp[0]:
            wamid = messages_resp[0]["id"]
            logger.info("[META] Message accepted by Meta")
            logger.info("[META] WhatsApp message ID: %s", wamid)
        else:
            logger.info("[META] Message accepted by Meta, response: %s", response.text)
        return resp_json
    except Exception:
        logger.info("[META] Response body: %s", response.text)
        return response.json()

# ---------------------------------------------------------------------------
# AI agent processing
# ---------------------------------------------------------------------------
def process_whatsapp_message(sender_id, incoming_msg, message_id):
    """Run the incoming message through the Groq AI agent and reply via Meta."""
    print("[PIPELINE] -- START process_whatsapp_message --", flush=True)
    logger.info("[PIPELINE] -- START process_whatsapp_message --")
    logger.info("[PIPELINE] ── START id=%s from=%s*** ──", message_id, str(sender_id)[:4])

    print("[PIPELINE] Extracted message text", flush=True)
    logger.info("[INPUT DEBUG] text=%s", incoming_msg)
    logger.info("[PIPELINE] Extracted message text (len=%d)", len(incoming_msg))
    logger.info("[PIPELINE] Sender ID prefix: %s***", str(sender_id)[:4])

    # ── Stage 1: Supabase client init ────────────────────────────────────────
    print("[PIPELINE] Stage 1 starting", flush=True)
    logger.info("[PIPELINE] Stage 1 starting — Initialising Supabase client")
    try:
        supabase = get_supabase()
        print("[PIPELINE] Stage 1 completed", flush=True)
        logger.info("[PIPELINE] Stage 1 completed — Supabase client OK")
    except Exception as e:
        print("[PIPELINE] Stage 1 FAILED", flush=True)
        traceback.print_exc()
        logger.exception("[PIPELINE] Stage 1 FAILED — could not create Supabase client")
        _send_error_reply(sender_id)
        return False

    # ── Stage 2: CRM history lookup ───────────────────────────────────────────
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
    except Exception as e:
        print("[PIPELINE] Stage 2 FAILED", flush=True)
        traceback.print_exc()
        logger.exception("[PIPELINE] Stage 2 FAILED — Supabase history fetch error")
        raw_history = []

    # Build clean alternating history list (chronological order)
    chronological_history = list(reversed(raw_history))
    cleaned_history = []
    for h in chronological_history:
        role = h.get("role")
        content = h.get("message_text", "").strip()
        if role in ("user", "assistant") and content:
            # Avoid consecutive duplicate messages of the same role in history
            if cleaned_history and cleaned_history[-1]["role"] == role and cleaned_history[-1]["content"] == content:
                continue
            cleaned_history.append({"role": role, "content": content})

    # If the last item in history is a user message, drop it so it doesn't collide with the current incoming user turn
    if cleaned_history and cleaned_history[-1]["role"] == "user":
        cleaned_history.pop()

    last_role = cleaned_history[-1]["role"] if cleaned_history else "none"
    last_msg_snippet = cleaned_history[-1]["content"][:60] if cleaned_history else "none"

    logger.info("[LLM DEBUG] current_message=%s", incoming_msg)
    logger.info("[LLM DEBUG] history_count=%d", len(cleaned_history))
    logger.info("[LLM DEBUG] history_last_role=%s", last_role)
    logger.info("[LLM DEBUG] history_last_message=%s", last_msg_snippet)

    # Build message list for LLM
    messages = [
        {
            "role": "system",
            "content": (
                "You are an intelligent construction field ops assistant. Answer queries concisely for WhatsApp output. Use bold formatting where appropriate.\n\n"
                "CRITICAL GROUNDING RULES:\n"
                "1. For greetings, pleasantries, or general conversational openers (e.g. 'Hello', 'Hi', 'Hey', 'Help', 'Who are you?'), respond politely and warmly WITHOUT calling any database tools.\n"
                "2. When the user asks about projects (e.g. 'What are my active projects?', 'List projects', 'ongoing projects'), daily progress, site logs, or expenses, ALWAYS invoke the appropriate database tool to query Supabase.\n"
                "3. Your final answer MUST be strictly and exclusively grounded in the data returned by the database tools. NEVER invent, assume, or hallucinate project names, locations, metrics, or expenses (e.g., only mention projects returned in the tool response).\n"
                "4. If a tool returns 0 projects or 0 records, clearly state 'No active projects found.' or 'No records exist.' Never make up fictitious project names or data."
            )
        }
    ]
    messages.extend(cleaned_history)
    messages.append({"role": "user", "content": incoming_msg})
    logger.info("[PIPELINE] LLM context built: %d message(s) (incl. system)", len(messages))

    # ── Stage 3: Persist user message ────────────────────────────────────────
    print("[PIPELINE] Stage 3 starting", flush=True)
    logger.info("[PIPELINE] Stage 3 starting — Persisting user message to Supabase")
    try:
        supabase.table("chat_sessions").insert({
            "whatsapp_number": sender_id,
            "role": "user",
            "message_text": incoming_msg
        }).execute()
        print("[PIPELINE] Stage 3 completed", flush=True)
        logger.info("[PIPELINE] Stage 3 completed — User message persisted OK")
    except Exception as e:
        print("[PIPELINE] Stage 3 FAILED", flush=True)
        traceback.print_exc()
        logger.exception("[PIPELINE] Stage 3 FAILED — Supabase insert user message error")
        # Non-fatal: continue even if logging to DB fails

    # ── Stage 4: Groq client init ─────────────────────────────────────────────
    print("[PIPELINE] Stage 4 starting", flush=True)
    logger.info("[PIPELINE] Stage 4 starting — Initialising Groq client")
    try:
        groq_client = get_groq()
        print("[PIPELINE] Stage 4 completed", flush=True)
        logger.info("[PIPELINE] Stage 4 completed — Groq client OK")
    except Exception as e:
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
            timeout=25  # seconds — prevents indefinite hang
        )
        print("[PIPELINE] Stage 5 completed", flush=True)
        logger.info("[PIPELINE] Stage 5 completed — LLM call completed")
    except Exception as e:
        print("[PIPELINE] Stage 5 FAILED", flush=True)
        traceback.print_exc()
        logger.exception("[PIPELINE] Stage 5 FAILED — Groq first LLM call error")
        _send_error_reply(sender_id)
        return False

    msg_obj = response.choices[0].message
    has_tool_calls = bool(msg_obj.tool_calls)
    logger.info("[TOOL DEBUG] called=%s", str(has_tool_calls).lower())
    logger.info("[TOOL DEBUG] tool_calls=%s", str(has_tool_calls).lower())
    logger.info("[PIPELINE] Stage 5 — finish_reason=%s tool_calls=%s",
                response.choices[0].finish_reason,
                has_tool_calls)

    # ── Stage 6: Tool execution (if requested) ────────────────────────────────
    if has_tool_calls:
        print("[PIPELINE] Stage 6 starting", flush=True)
        logger.info("[PIPELINE] Stage 6 starting")
        tool_names = [tc.function.name for tc in msg_obj.tool_calls]
        logger.info("[PIPELINE] Stage 6 — Tool calls requested: %s", tool_names)

        messages.append({
            "role": "assistant",
            "content": msg_obj.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
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
                logger.error("[PIPELINE] Stage 6 — JSON decode error on tool args for %s",
                             tool_name)
                tool_args = {}

            logger.info("[TOOL DEBUG] name=%s", tool_name)
            logger.info("[TOOL DEBUG] reason=model requested execution for query: %s", incoming_msg[:40])
            logger.info("[TOOL DEBUG] arguments=%s", tool_args)
            logger.info("[PIPELINE] Stage 6 — Executing tool: %s", tool_name)
            try:
                tool_output = execute_tool(tool_name, tool_args)
                logger.info("[PIPELINE] Stage 6 — Tool %s completed, output_len=%d",
                            tool_name, len(tool_output))
            except Exception as e:
                print(f"[PIPELINE] Stage 6 tool {tool_name} FAILED", flush=True)
                traceback.print_exc()
                logger.exception("[PIPELINE] Stage 6 FAILED — tool %s raised exception",
                                 tool_name)
                tool_output = json.dumps({"error": "Tool execution failed."})

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_output
            })

        print("[PIPELINE] Stage 6 completed", flush=True)
        logger.info("[PIPELINE] Stage 6 completed")

        # ── Stage 7: Second LLM call (final answer after tools) ───────────────
        print("[PIPELINE] Stage 7 starting", flush=True)
        logger.info("[PIPELINE] Stage 7 starting — Starting second LLM call (after tools)")
        try:
            second_response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                timeout=25
            )
            reply_text = second_response.choices[0].message.content
            print("[PIPELINE] Stage 7 completed", flush=True)
            logger.info("[PIPELINE] Stage 7 completed — Second LLM call completed")
        except Exception as e:
            print("[PIPELINE] Stage 7 FAILED", flush=True)
            traceback.print_exc()
            logger.exception("[PIPELINE] Stage 7 FAILED — Groq second LLM call error")
            _send_error_reply(sender_id)
            return False
    else:
        print("[PIPELINE] Stage 6 starting", flush=True)
        print("[PIPELINE] Stage 6 completed", flush=True)
        logger.info("[PIPELINE] Stage 6 — No tool calls, using direct LLM answer")
        reply_text = msg_obj.content

    print("[PIPELINE] Generated response", flush=True)
    logger.info("[PIPELINE] Generated response (len=%d chars)", len(reply_text or ""))

    # ── Stage 8: Persist assistant reply ─────────────────────────────────────
    print("[PIPELINE] Stage 8 starting", flush=True)
    logger.info("[PIPELINE] Stage 8 starting — Persisting assistant reply to Supabase")
    try:
        supabase.table("chat_sessions").insert({
            "whatsapp_number": sender_id,
            "role": "assistant",
            "message_text": reply_text
        }).execute()
        print("[PIPELINE] Stage 8 completed", flush=True)
        logger.info("[PIPELINE] Stage 8 completed — Assistant reply persisted OK")
    except Exception as e:
        print("[PIPELINE] Stage 8 FAILED", flush=True)
        traceback.print_exc()
        logger.exception("[PIPELINE] Stage 8 FAILED — Supabase insert assistant message error")
        # Non-fatal: still attempt to send the reply

    # ── Stage 9: Send reply via Meta ──────────────────────────────────────────
    print("[PIPELINE] Stage 9 starting", flush=True)
    logger.info("[PIPELINE] Stage 9 starting — Starting Meta send")
    try:
        send_whatsapp_message(sender_id, reply_text)
        print("[PIPELINE] Stage 9 completed", flush=True)
        logger.info("[PIPELINE] Stage 9 completed ✅")
    except Exception as e:
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
            "⚠️ Sorry, I encountered an error processing your request. Please try again."
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
    # -----------------------------------------------------------------------
    # DIAGNOSTIC BANNER — appears in Vercel logs for every POST received.
    # -----------------------------------------------------------------------
    print("=== META WHATSAPP WEBHOOK POST RECEIVED ===", flush=True)
    logger.info("=== META WHATSAPP WEBHOOK POST RECEIVED ===")
    logger.info("[WEBHOOK] POST received")
    logger.info("[WEBHOOK] method=%s path=%s content_type=%s",
                request.method, request.path, request.content_type)

    # Parse body
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
                field = change.get("field", "unknown")
                value = change.get("value", {})
                statuses = value.get("statuses", [])
                msgs = value.get("messages", [])

                logger.info("[WEBHOOK] field=%s", field)
                logger.info("[WEBHOOK] message_event=%s", str(bool(msgs)).lower())

                # Status / delivery / read events — acknowledge and skip
                if statuses:
                    for status in statuses:
                        logger.info(
                            "[WEBHOOK] Status event: id=%s status=%s",
                            status.get("id"), status.get("status")
                        )
                    continue

                # Incoming messages
                if not msgs:
                    logger.info("[WEBHOOK] No messages in this change entry, skipping")
                    continue

                msg        = msgs[0]
                message_id = msg.get("id", "unknown")
                sender_id  = msg.get("from", "")
                msg_type   = msg.get("type", "unknown")

                # Mask sender for safe logging (show first 4 digits only)
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
                except Exception as e:
                    print("[PIPELINE] EXCEPTION while calling process_whatsapp_message():", e, flush=True)
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