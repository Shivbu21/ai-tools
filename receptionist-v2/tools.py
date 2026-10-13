"""Local JSON-backed tool implementations for the receptionist.

All state lives in plain JSON files next to this module:
  slots.json          - appointment grid; booking marks a slot taken (no double-booking)
  messages.json       - caller messages for the owner
  confirmations.json  - recorded "confirmation" messages (offline: recorded + spoken aloud,
                        NOT sent anywhere - labeled honestly in the schema and README)

Every function returns a dict: {"ok": bool, "say": str, "data": {...}} where "say"
is what the receptionist should read aloud to the caller.
"""
import json
import os
import re
import threading
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.abspath(__file__))
SLOTS_FILE = os.path.join(BASE, "slots.json")
MESSAGES_FILE = os.path.join(BASE, "messages.json")
CONFIRMATIONS_FILE = os.path.join(BASE, "confirmations.json")

_lock = threading.Lock()


def _load(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _save(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _rows(path):
    data = _load(path)
    return data if isinstance(data, list) else []


def check_availability(args):
    date = (args.get("date") or "").strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        return {"ok": False, "say": f"I couldn't understand the date '{date}'. Please give it as year-month-day.",
                "data": {"date": date, "free": []}}
    with _lock:
        store = _load(SLOTS_FILE) or {"slots": []}
        free = sorted(s["time"] for s in store.get("slots", [])
                      if s.get("date") == date and not s.get("taken"))
    if not free:
        return {"ok": False, "say": f"Sorry, nothing is free on {date}. Would you like a different day?",
                "data": {"date": date, "free": []}}
    return {"ok": True, "say": f"On {date} I have {', '.join(free)} free.",
            "data": {"date": date, "free": free}}


def book_slot(args):
    name = (args.get("name") or "").strip()
    phone = re.sub(r"\D", "", str(args.get("phone") or ""))
    date = (args.get("date") or "").strip()
    time_ = (args.get("time") or "").strip()
    if not name or len(phone) < 10 or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date) \
            or not re.fullmatch(r"\d{2}:\d{2}", time_):
        return {"ok": False, "say": "I need your name, a 10-digit phone number, and a valid date and time to book.",
                "data": {}}
    with _lock:
        store = _load(SLOTS_FILE) or {"slots": []}
        target = None
        for s in store.get("slots", []):
            if s.get("date") == date and s.get("time") == time_:
                target = s
                break
        if target is None:
            return {"ok": False,
                    "say": f"There is no {time_} slot on {date}. Ask me to check availability for another time.",
                    "data": {"date": date, "time": time_}}
        if target.get("taken"):
            # find an alternative on the same day to suggest
            alt = next((s["time"] for s in store.get("slots", [])
                        if s.get("date") == date and not s.get("taken")), None)
            hint = f" {alt} on the same day is still free." if alt else ""
            return {"ok": False,
                    "say": f"Sorry, {time_} on {date} is already booked.{hint}",
                    "data": {"date": date, "time": time_, "already_taken": True, "alt": alt}}
        target["taken"] = True
        target["by"] = {"name": name, "phone": phone,
                        "booked_at": datetime.now(timezone.utc).isoformat()}
        _save(SLOTS_FILE, store)
    return {"ok": True,
            "say": f"Done, {name}! I've booked {date} at {time_} for you. We'll confirm on {phone}.",
            "data": {"date": date, "time": time_, "name": name, "phone": phone}}


def take_message(args):
    frm = (args.get("from") or "").strip()
    phone = re.sub(r"\D", "", str(args.get("phone") or ""))
    message = (args.get("message") or "").strip()
    if not message:
        return {"ok": False, "say": "What message should I pass on to the owner?",
                "data": {}}
    with _lock:
        rows = _rows(MESSAGES_FILE)
        rows.append({"ts": datetime.now(timezone.utc).isoformat(), "from": frm,
                     "phone": phone, "message": message})
        _save(MESSAGES_FILE, rows)
    return {"ok": True,
            "say": "Noted. I've written your message down and the owner will see it first thing.",
            "data": {"from": frm, "message": message}}


def send_confirmation(args):
    phone = re.sub(r"\D", "", str(args.get("phone") or ""))
    text = (args.get("text") or "").strip()
    if len(phone) < 10 or not text:
        return {"ok": False, "say": "I need a 10-digit phone number and a confirmation message.",
                "data": {}}
    with _lock:
        rows = _rows(CONFIRMATIONS_FILE)
        rows.append({"ts": datetime.now(timezone.utc).isoformat(), "phone": phone,
                     "text": text, "sent": False,
                     "note": "offline build: recorded locally, not actually sent via SMS"})
        _save(CONFIRMATIONS_FILE, rows)
    return {"ok": True,
            "say": "I've recorded your confirmation and will read it back to you now.",
            "data": {"phone": phone, "text": text, "sent": False}}


TOOLS = {
    "check_availability": check_availability,
    "book_slot": book_slot,
    "take_message": take_message,
    "send_confirmation": send_confirmation,
}


def run_tool(name, args, transcript=""):
    fn = TOOLS.get(name)
    if fn is None:
        return {"ok": False, "say": "I don't have a tool for that, sorry.",
                "data": {"unknown_tool": name}}
    if not isinstance(args, dict):
        args = {}
    try:
        return fn(args)
    except Exception as e:  # never crash the call on a tool bug
        return {"ok": False, "say": "Something went wrong running that. Let's try another way.",
                "data": {"error": str(e)}}
