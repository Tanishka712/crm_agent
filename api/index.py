import os
import json
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from supabase import create_client, Client as SupabaseClient
from groq import Groq
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Initialize Clients
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

supabase: SupabaseClient = create_client(SUPABASE_URL, SUPABASE_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

GROQ_MODEL = "llama-3.3-70b-versatile"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_project_summary",
            "description": "Fetch overall progress, expenses, and equipment stats for a given project name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_name": {"type": "string", "description": "The name of the project."}
                },
                "required": ["project_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_expenses",
            "description": "Get site expenses for a project or all projects.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 5}
                }
            }
        }
    }
]

def execute_tool(name, args):
    if name == "get_project_summary":
        p_name = args.get("project_name")
        res = supabase.table("projects").select("*").ilike("project_name", f"%{p_name}%").execute()
        if not res.data:
            return json.dumps({"error": f"No project found matching '{p_name}'"})
        
        project_id = res.data[0]["id"]
        p_real_name = res.data[0]["project_name"]

        logs = supabase.table("daily_logs").select("*").eq("project_id", project_id).execute().data or []
        expenses = supabase.table("expenses").select("*").eq("project_id", project_id).execute().data or []
        equip = supabase.table("equipment_logs").select("*").eq("project_id", project_id).execute().data or []

        total_trenching = sum([l.get("trenching_meters", 0) or 0 for l in logs])
        total_pipe = sum([l.get("pipe_laid_meters", 0) or 0 for l in logs])
        total_spent = sum([e.get("amount", 0) or 0 for e in expenses])

        return json.dumps({
            "project_name": p_real_name,
            "total_trenching_meters": total_trenching,
            "total_pipe_laid_meters": total_pipe,
            "total_expenses_inr": total_spent,
            "equipment_logs_count": len(equip)
        })

    elif name == "get_recent_expenses":
        limit = args.get("limit", 5)
        res = supabase.table("expenses").select("category, amount, payment_mode, vendor_or_recipient, expense_date").order("expense_date", desc=True).limit(limit).execute()
        return json.dumps(res.data or [])
    
    return json.dumps({"error": "Unknown tool function."})

def process_whatsapp_message(from_number, incoming_msg):
    try:
        logging.info("Processing WhatsApp message from %s", from_number)
        history_res = supabase.table("chat_sessions").select("role, message_text").eq("whatsapp_number", from_number).order("created_at", desc=True).limit(6).execute()

        messages = [{"role": "system", "content": "You are an intelligent construction field ops assistant. Answer queries concisely for WhatsApp output. Use bold formatting where appropriate."}]

        for h in reversed(history_res.data or []):
            if h["role"] in ["user", "assistant"]:
                messages.append({"role": h["role"], "content": h["message_text"]})

        messages.append({"role": "user", "content": incoming_msg})
        supabase.table("chat_sessions").insert({"whatsapp_number": from_number, "role": "user", "message_text": incoming_msg}).execute()

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )

        msg_obj = response.choices[0].message

        if msg_obj.tool_calls:
            messages.append(msg_obj)

            for tool_call in msg_obj.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                tool_output = execute_tool(tool_name, tool_args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_output
                })

            second_response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages
            )
            reply_text = second_response.choices[0].message.content
        else:
            reply_text = msg_obj.content

        supabase.table("chat_sessions").insert({"whatsapp_number": from_number, "role": "assistant", "message_text": reply_text}).execute()

        twilio_client.messages.create(
            from_=os.getenv("TWILIO_WHATSAPP_NUMBER"),
            to=from_number,
            body=reply_text
        )
        return True

    except Exception as e:
        logging.exception("Failed to process WhatsApp message")
        try:
            supabase.table("chat_sessions").insert({
                "whatsapp_number": from_number,
                "role": "error",
                "message_text": f"Processing error: {e}"
            }).execute()

            twilio_client.messages.create(
                from_=os.getenv("TWILIO_WHATSAPP_NUMBER"),
                to=from_number,
                body="⚠️ Sorry, I encountered an error processing your request. Please try again."
            )
        except Exception:
            logging.exception("Failed to write error or send message")
        return False

@app.route("/webhook/whatsapp", methods=["GET", "POST"])
def whatsapp_webhook():
    if request.method == "GET":
        return "WhatsApp Webhook Endpoint is Active!", 200

    from_number = request.values.get("From", "")
    incoming_msg = request.values.get("Body", "").strip()

    # Synchronous processing suitable for Vercel Serverless Functions
    process_whatsapp_message(from_number, incoming_msg)

    resp = MessagingResponse()
    return str(resp)

@app.route("/debug/chat_sessions", methods=["GET"])
def debug_chat_sessions():
    try:
        res = supabase.table("chat_sessions").select("*").order("created_at", desc=True).limit(10).execute()
        return {"count": len(res.data or []), "recent": res.data}
    except Exception as e:
        logging.exception("Debug chat_sessions failed")
        return {"error": str(e)}, 500