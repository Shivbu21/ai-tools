# Receptionist v2 — agentic voice receptionist (offline)

Extends `~/workspace/pixalified-voice-agent/` with **real tool calling**: the
1.5B model emits JSON tool calls, they execute against local JSON stores, the
result feeds back into the conversation, and the outcome is spoken via Piper.

## Architecture

```
browser mic (webm) → ffmpeg → wav 16k → faster-whisper tiny (STT)
  → qwen2.5-1.5b via llama-server (agentic loop, JSON tool calls)
  → tools.py executes: slots.json / messages.json / confirmations.json
  → tool result fed back → final spoken reply
  → Piper TTS → wav to browser
```

Files:

| file | role |
|---|---|
| `server.py` | FastAPI: `POST /api/turn` (voice), `POST /api/text-turn` (text, no mic), widget at `/` |
| `brain.py` | Agentic loop: system prompt with tool defs + today's date, JSON parsing, up to 3 tool rounds |
| `tools.py` | Tool implementations with a lock; **no double-booking enforced here** |
| `tools.json` | Tool schemas fed to the model |
| `slots.json` | Weekly availability grid (seeded 2026-10-15→21; two slots pre-taken) |
| `messages.json`, `confirmations.json` | Caller messages / recorded confirmations (created on use) |
| `test_turn.py` | Scripted e2e test, no mic |

## Run

```bash
# 1. LLM (one shared server for both ai-tools projects)
~/llamacpp/build/bin/llama-server \
  -m /home/hatch/ollama/models/blobs/sha256-183715c435899236895da3869489cc30ac241476b4971a20285b1a462818a5b4 \
  -t 2 -c 2048 --port 11435

# 2. receptionist server (reuses the pixalified-voice-agent venv)
~/workspace/pixalified-voice-agent/venv/bin/python ~/workspace/ai-tools/receptionist-v2/server.py
# → http://127.0.0.1:8766  (widget: hold-to-talk voice UI)

# 3. scripted test (no mic)
~/workspace/pixalified-voice-agent/venv/bin/python ~/workspace/ai-tools/receptionist-v2/test_turn.py
```

## Tools (all local, all real)

- `check_availability(date)` — reads `slots.json`, returns free times.
- `book_slot(name, phone, date, time)` — marks a slot taken; **rejects if already
  taken or nonexistent** (verified by test).
- `take_message(from, phone, message)` — appends to `messages.json`.
- `send_confirmation(phone, text)` — appends to `confirmations.json` with
  `"sent": false`. **Honest label: nothing is actually sent** — this is an
  offline build; "sending" = recorded + read aloud.

## Verified

`test_turn.py` (2026-10-14): utterance *"book me a slot tomorrow at 5pm, Priya
Sharma, 9812345678"* → model emitted
`{"tool":"book_slot","args":{"name":"Priya Sharma","phone":"9812345678","date":"2026-10-15","time":"17:00"}}`
→ `slots.json` recorded the booking (taken=true, correct name/phone) →
double-booking the same slot was rejected and the original booking was
preserved → Piper produced a real WAV reply (`test_reply.wav`). **PASS.**

## Honest limitations

- **Tool-use reliability of a 1.5B model is the weak link.** It follows the
  schema for simple booking requests, but it can emit malformed JSON, resolve
  dates wrong ("Friday" on a wrong week), or hallucinate a confirmation instead
  of calling the tool. `tools.py` validates everything server-side, so a bad
  call fails safely instead of corrupting state — but expect misses on
  multi-step or ambiguous requests. A 7B+ model would be meaningfully better.
- **No real SMS/email sending.** `send_confirmation` records locally and the
  reply is spoken; the record is explicitly marked `"sent": false`.
- Whisper tiny mishears names/phone numbers sometimes — the brain asks for
  clarification, which costs a turn.
- Latency: ~9–15s per turn on 2 CPU cores (STT + LLM + TTS).
- Single-process file lock only; concurrent multi-caller use needs a real DB.
