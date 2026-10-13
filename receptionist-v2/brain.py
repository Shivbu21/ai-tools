"""LLM brain for receptionist-v2: agentic loop around the local qwen2.5-1.5b.

The model is asked to emit JSON tool calls; each call is executed against the
local JSON stores in tools.py, the result is fed back, and the loop repeats
until the model produces plain speech text (max 3 tool rounds).

Honest caveat: a 1.5B model is a weak tool-user. It usually follows the schema
for simple requests, but it can emit malformed JSON or wrong dates. parse_tool_call
tolerates code fences; the loop retries once with a stricter nudge.
"""
import json
import os
import re

os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

import httpx

from tools import run_tool

BASE = os.path.dirname(os.path.abspath(__file__))
LLM_API_URL = os.environ.get("LLM_API_URL", "http://127.0.0.1:11435/v1").rstrip("/")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5-1.5b")

with open(os.path.join(BASE, "tools.json")) as f:
    TOOL_DEFS = json.load(f)["tools"]


def _tool_block():
    lines = ["Available tools. To use one, output ONLY a single JSON object on one line:",
             '{"tool":"<name>","args":{...}}',
             "Nothing else on that line. When you are done with tools, reply in plain conversational text."]
    for t in TOOL_DEFS:
        lines.append(f"- {t['name']}{t['args']}: {t['description']}")
    return "\n".join(lines)


def build_system_prompt(today_str):
    return f"""You are Maya, AI voice receptionist for a small shop in India (IST).
Today is {today_str}. Slots are booked in Asia/Kolkata.
Rules:
- Be warm and brief: reply in under 35 words when speaking to the caller.
- If the caller wants to book: get their name and 10-digit phone number, then book. Convert relative dates to YYYY-MM-DD using today's date.
- For price or offer questions: say pricing depends on requirements and offer an owner callback.
{_tool_block()}
Examples:
caller: book me a slot tomorrow at 5pm, I'm Priya 9812345678
you: {{"tool":"book_slot","args":{{"name":"Priya","phone":"9812345678","date":"2026-10-15","time":"17:00"}}}}
caller: what is free on Friday
you: {{"tool":"check_availability","args":{{"date":"2026-10-16"}}}}
"""


def _post(messages, max_tokens):
    r = httpx.post(
        f"{LLM_API_URL}/chat/completions",
        json={"model": LLM_MODEL, "messages": messages,
              "max_tokens": max_tokens, "temperature": 0.2},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def parse_tool_call(text):
    """Extract {"tool":..., "args":{...}} from model output. Returns (name, args) or None."""
    t = text.strip().strip("`")
    m = re.search(r"\{[^{}]*\"tool\"\s*:\s*\"[a-z_]+\"[^{}]*\}", t, re.DOTALL)
    candidate = m.group(0) if m else t
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and isinstance(obj.get("tool"), str) and isinstance(obj.get("args"), dict):
        return obj["tool"], obj["args"]
    return None


def agentic_turn(transcript, today_str, max_rounds=3):
    """Run one conversational turn. Returns dict with reply_text, transcript, tool_calls, spoken_text."""
    messages = [
        {"role": "system", "content": build_system_prompt(today_str)},
        {"role": "user", "content": transcript},
    ]
    tool_calls = []
    for _ in range(max_rounds):
        raw = _post(messages, max_tokens=150)
        parsed = parse_tool_call(raw)
        if not parsed:
            return {"transcript": transcript, "reply_text": raw,
                    "tool_calls": tool_calls, "spoken_text": raw}
        name, args = parsed
        result = run_tool(name, args, transcript)
        tool_calls.append({"name": name, "args": args, "ok": result["ok"]})
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content":
                         f"[tool {name} returned ok={result['ok']}: {json.dumps(result['data'])}] "
                         "Now reply to the caller in plain conversational text under 35 words. "
                         "Do not output JSON again."})
        # one final short generation for the spoken reply
        final = _post(messages, max_tokens=90)
        if parse_tool_call(final):
            # model tried to call another tool anyway; use the tool's own "say"
            final = result["say"]
        return {"transcript": transcript, "reply_text": final,
                "tool_calls": tool_calls, "spoken_text": final}
    # fell through: use last tool result's spoken line
    last_say = tool_calls[-1] if tool_calls else None
    spoken = run_tool(last_say["name"], last_say["args"])["say"] if last_say else "Sorry, I couldn't complete that."
    return {"transcript": transcript, "reply_text": spoken,
            "tool_calls": tool_calls, "spoken_text": spoken}
