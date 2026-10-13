"""Scripted end-to-end test for receptionist-v2 (NO mic needed).

Flow: text utterance -> brain (agentic tool loop) -> tools.py executes
-> Piper TTS synthesis -> assertions.

Asserts:
  1. book_slot tool fires and returns ok=True
  2. slots.json actually records the booking (no fake AI)
  3. double-booking the same slot is rejected
  4. a real reply WAV file is produced (RIFF header, nonzero)
Run with the shared voice venv:
  ~/workspace/pixalified-voice-agent/venv/bin/python test_turn.py
"""
import io
import json
import os
import shutil
import sys
import wave
from datetime import datetime
from zoneinfo import ZoneInfo

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
os.environ["LLM_API_URL"] = "http://127.0.0.1:11435/v1"

import numpy as np

from brain import agentic_turn
from tools import SLOTS_FILE

# --- reset stores to a known state (backup first) ---
BACKUP = {}
for name in ("slots.json", "messages.json", "confirmations.json"):
    p = os.path.join(BASE, name)
    if os.path.exists(p):
        BACKUP[name] = p + ".testbak"
        shutil.copy(p, BACKUP[name])

with open(SLOTS_FILE) as f:
    seed = json.load(f)

TODAY = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %A")
TARGET_DATE = "2026-10-15"  # "tomorrow" on 2026-10-14 IST
TARGET_TIME = "17:00"
UTTERANCE = ("Hi, I'd like to book a slot tomorrow at 5pm. "
             "My name is Priya Sharma, phone 9812345678.")

print("utterance:", UTTERANCE, flush=True)
turn = agentic_turn(UTTERANCE, TODAY)
print("tool_calls:", json.dumps(turn["tool_calls"], indent=2), flush=True)
print("reply:     ", turn["reply_text"], flush=True)

# --- assertions ---
assert turn["tool_calls"], "brain emitted no tool call at all"
book = next((c for c in turn["tool_calls"] if c["name"] == "book_slot"), None)
assert book, f"expected a book_slot call, got {[c['name'] for c in turn['tool_calls']]}"
assert book["ok"], f"book_slot failed: {book}"
assert book["args"].get("date") == TARGET_DATE, f"wrong date resolved: {book['args']}"
assert book["args"].get("time") == TARGET_TIME, f"wrong time resolved: {book['args']}"

with open(SLOTS_FILE) as f:
    slots = json.load(f)["slots"]
row = next(s for s in slots if s["date"] == TARGET_DATE and s["time"] == TARGET_TIME)
assert row["taken"] is True, "slot not marked taken in slots.json"
assert row["by"]["name"].lower().startswith("priya"), f"wrong booker: {row['by']}"
assert "9812345678" in row["by"]["phone"], f"wrong phone: {row['by']}"
print("slots.json booking verified:", json.dumps(row), flush=True)

# double-booking must be rejected
turn2 = agentic_turn("Book me the same slot again, tomorrow at 5pm. I'm Rahul, 9822001122.", TODAY)
book2 = next((c for c in turn2["tool_calls"] if c["name"] == "book_slot"), None)
assert book2 is not None, "no book_slot call on second attempt"
assert book2["ok"] is False, f"DOUBLE-BOOKING ACCEPTED: {book2}"
with open(SLOTS_FILE) as f:
    row2 = next(s for s in json.load(f)["slots"]
                if s["date"] == TARGET_DATE and s["time"] == TARGET_TIME)
assert row2["by"]["name"].lower().startswith("priya"), "double-booking overwrote the slot!"
print("double-booking correctly rejected:", turn2["reply_text"], flush=True)

# TTS: synthesize the spoken reply to a real WAV
VOICE_BASE = "/home/hatch/workspace/pixalified-voice-agent"
from piper import PiperVoice
voice = PiperVoice.load(os.path.join(VOICE_BASE, "models", "piper", "en_US-amy-medium.onnx"))
chunks = [(c.audio_float_array * 32767).astype(np.int16) for c in voice.synthesize(turn["spoken_text"])]
pcm = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
wav_path = os.path.join(BASE, "test_reply.wav")
with wave.open(wav_path, "wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(voice.config.sample_rate)
    wf.writeframes(pcm.tobytes())
size = os.path.getsize(wav_path)
with open(wav_path, "rb") as f:
    assert f.read(4) == b"RIFF", "reply is not a real WAV"
assert size > 10000, f"reply WAV suspiciously small: {size} bytes"
print(f"reply audio: {wav_path} ({size//1024} KB WAV), {len(pcm)/voice.config.sample_rate:.1f}s", flush=True)

# --- restore backups ---
for name, bak in BACKUP.items():
    shutil.copy(bak, os.path.join(BASE, name))
    os.unlink(bak)
print("stores restored; slots.json/test data unchanged from before the run", flush=True)
print("TEST_TURN PASS", flush=True)
