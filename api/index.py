import os
import json
import requests
from flask import Flask, request, Response
from supabase import create_client, Client as SupabaseClient
from groq import Groq
from dotenv import load_dotenv
import logging
import threading

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
            "name": "get_project_summary",
            "description": "Fetch overall progress, expenses, and equipment stats for a given project name.",
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
            "name": "get_recent_expenses",
            "description": "Get recent site expenses. The limit must be a whole number between 1 and 50.",
            "parameters": {
                "type": "object",
                "properties": {
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

    if name == "get_project_summary":
        p_name = args.get("project_name")
        try:
            res = supabase.table("projects").select("*").ilike("project_name", f"%{p_name}%").execute()
        except Exception:
            logger.exception("Supabase error in get_project_summary")
            return json.dumps({"error": "Database error fetching project."})

        if not res.data:
            return json.dumps({"error": f"No project found matching '{p_name}'"})

        project_id = res.data[0]["id"]
        p_real_name = res.data[0]["project_name"]

        try:
            logs    = supabase.table("daily_logs").select("*").eq("project_id", project_id).execute().data or []
            expenses = supabase.table("expenses").select("*").eq("project_id", project_id).execute().data or []
            equip   = supabase.table("equipment_logs").select("*").eq("project_id", project_id).execute().data or []
        except Exception:
            logger.exception("Supabase error fetching project sub-tables")
            return json.dumps({"error": "Database error fetching project details."})

        total_trenching = sum(l.get("trenching_meters", 0) or 0 for l in logs)
        total_pipe      = sum(l.get("pipe_laid_meters", 0) or 0 for l in logs)
        total_spent     = sum(e.get("amount", 0) or 0 for e in expenses)

        return json.dumps({
            "project_name": p_real_name,
            "total_trenching_meters": total_trenching,
            "total_pipe_laid_meters": total_pipe,
            "total_expenses_inr": total_spent,
            "equipment_logs_count": len(equip)
        })

    elif name == "get_recent_expenses":
        limit = args.get("limit", 5)
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 5
        limit = max(1, min(limit, 50))

        try:
            res = supabase.table("expenses").select(
                "category, amount, payment_mode, vendor_or_recipient, expense_date"
            ).order("expense_date", desc=True).limit(limit).execute()
        except Exception:
            logger.exception("Supabase error in get_recent_expenses")
            return json.dumps({"error": "Database error fetching expenses."})

        return json.dumps(res.data)

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
    logger.info("[PIPELINE] ── START id=%s from=%s*** ──", message_id, str(sender_id)[:4])
    logger.info("[PIPELINE] Extracted message text (len=%d)", len(incoming_msg))
    logger.info("[PIPELINE] Sender ID prefix: %s***", str(sender_id)[:4])

    # ── Stage 1: Supabase client init ────────────────────────────────────────
    logger.info("[PIPELINE] Stage 1 — Initialising Supabase client")
    try:
        supabase = get_supabase()
        logger.info("[PIPELINE] Stage 1 — Supabase client OK")
    except Exception:
        logger.exception("[PIPELINE] Stage 1 FAILED — could not create Supabase client")
        _send_error_reply(sender_id)
        return False

    # ── Stage 2: CRM history lookup ───────────────────────────────────────────
    logger.info("[PIPELINE] Stage 2 — Starting CRM history lookup")
    try:
        history_res = (
            supabase.table("chat_sessions")
            .select("role, message_text")
            .eq("whatsapp_number", sender_id)
            .order("created_at", desc=True)
            .limit(6)
            .execute()
        )
        logger.info("[PIPELINE] Stage 2 — CRM lookup completed, rows=%d",
                    len(history_res.data or []))
    except Exception:
        logger.exception("[PIPELINE] Stage 2 FAILED — Supabase history fetch error")
        history_res = type("R", (), {"data": []})()

    # Build message list for LLM
    messages = [
        {
            "role": "system",
            "content": (
                "You are an intelligent construction field ops assistant. "
                "Answer queries concisely for WhatsApp output. "
                "Use bold formatting where appropriate."
            )
        }
    ]
    for h in reversed(history_res.data or []):
        if h["role"] in ("user", "assistant"):
            messages.append({"role": h["role"], "content": h["message_text"]})
    messages.append({"role": "user", "content": incoming_msg})
    logger.info("[PIPELINE] LLM context built: %d message(s) (incl. system)", len(messages))

    # ── Stage 3: Persist user message ────────────────────────────────────────
    logger.info("[PIPELINE] Stage 3 — Persisting user message to Supabase")
    try:
        supabase.table("chat_sessions").insert({
            "whatsapp_number": sender_id,
            "role": "user",
            "message_text": incoming_msg
        }).execute()
        logger.info("[PIPELINE] Stage 3 — User message persisted OK")
    except Exception:
        logger.exception("[PIPELINE] Stage 3 FAILED — Supabase insert user message error")
        # Non-fatal: continue even if logging to DB fails

    # ── Stage 4: Groq client init ─────────────────────────────────────────────
    logger.info("[PIPELINE] Stage 4 — Initialising Groq client")
    try:
        groq_client = get_groq()
        logger.info("[PIPELINE] Stage 4 — Groq client OK")
    except Exception:
        logger.exception("[PIPELINE] Stage 4 FAILED — could not create Groq client")
        _send_error_reply(sender_id)
        return False

    # ── Stage 5: First LLM call ───────────────────────────────────────────────
    logger.info("[PIPELINE] Stage 5 — Starting LLM call (model=%s, messages=%d)",
                GROQ_MODEL, len(messages))
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            timeout=25  # seconds — prevents indefinite hang
        )
        logger.info("[PIPELINE] Stage 5 — LLM call completed")
    except Exception:
        logger.exception("[PIPELINE] Stage 5 FAILED — Groq first LLM call error")
        _send_error_reply(sender_id)
        return False

    msg_obj = response.choices[0].message
    logger.info("[PIPELINE] Stage 5 — finish_reason=%s tool_calls=%s",
                response.choices[0].finish_reason,
                bool(msg_obj.tool_calls))

    # ── Stage 6: Tool execution (if requested) ────────────────────────────────
    if msg_obj.tool_calls:
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

            logger.info("[PIPELINE] Stage 6 — Executing tool: %s", tool_name)
            try:
                tool_output = execute_tool(tool_name, tool_args)
                logger.info("[PIPELINE] Stage 6 — Tool %s completed, output_len=%d",
                            tool_name, len(tool_output))
            except Exception:
                logger.exception("[PIPELINE] Stage 6 FAILED — tool %s raised exception",
                                 tool_name)
                tool_output = json.dumps({"error": "Tool execution failed."})

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_output
            })

        # ── Stage 7: Second LLM call (final answer after tools) ───────────────
        logger.info("[PIPELINE] Stage 7 — Starting second LLM call (after tools)")
        try:
            second_response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                timeout=25
            )
            reply_text = second_response.choices[0].message.content
            logger.info("[PIPELINE] Stage 7 — Second LLM call completed")
        except Exception:
            logger.exception("[PIPELINE] Stage 7 FAILED — Groq second LLM call error")
            _send_error_reply(sender_id)
            return False
    else:
        logger.info("[PIPELINE] Stage 6 — No tool calls, using direct LLM answer")
        reply_text = msg_obj.content

    logger.info("[PIPELINE] Generated response (len=%d chars)", len(reply_text or ""))

    # ── Stage 8: Persist assistant reply ─────────────────────────────────────
    logger.info("[PIPELINE] Stage 8 — Persisting assistant reply to Supabase")
    try:
        supabase.table("chat_sessions").insert({
            "whatsapp_number": sender_id,
            "role": "assistant",
            "message_text": reply_text
        }).execute()
        logger.info("[PIPELINE] Stage 8 — Assistant reply persisted OK")
    except Exception:
        logger.exception("[PIPELINE] Stage 8 FAILED — Supabase insert assistant message error")
        # Non-fatal: still attempt to send the reply

    # ── Stage 9: Send reply via Meta ──────────────────────────────────────────
    logger.info("[PIPELINE] Stage 9 — Starting Meta send")
    try:
        send_whatsapp_message(sender_id, reply_text)
        logger.info("[PIPELINE] Stage 9 — Meta send completed ✅")
    except Exception:
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

    IMPORTANT: We return HTTP 200 as quickly as possible (before processing)
    so Meta does not time out waiting for a response and retry the delivery.
    Processing is done in a background thread.
    """
    # -----------------------------------------------------------------------
    # DIAGNOSTIC BANNER — appears in Vercel logs for every POST received.
    # If this never appears, the request is not reaching the Vercel function.
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

    # -----------------------------------------------------------------------
    # Return HTTP 200 to Meta IMMEDIATELY.
    # Processing happens in a background thread so we don't block the response.
    # Meta requires a 200 within ~20 seconds; Vercel hobby timeout is 10s.
    # Synchronous LLM + DB calls easily exceed 10s → the webhook times out
    # without this pattern, which causes Meta to stop delivering messages.
    # -----------------------------------------------------------------------
    def _process_in_background():
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
                    process_whatsapp_message(sender_id, incoming_msg, message_id)

        except Exception:
            logger.exception("[WEBHOOK] Unexpected error in background processing")

    # Spawn background thread and return 200 immediately
    thread = threading.Thread(target=_process_in_background, daemon=True)
    thread.start()

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