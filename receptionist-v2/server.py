"""Receptionist v2: agentic voice receptionist with real local JSON tools.

Pipeline per turn:
  browser mic (webm) -> ffmpeg -> wav 16k mono -> faster-whisper tiny (STT)
  -> qwen2.5-1.5b via llama-server (agentic loop, JSON tool calls)
  -> tools.py executes against slots.json / messages.json / confirmations.json
  -> Piper TTS -> wav back to browser

Run:
  1. Start the shared LLM: ~/llamacpp/build/bin/llama-server -m <gguf> -t 2 -c 2048 --port 11435
  2. ~/workspace/pixalified-voice-agent/venv/bin/python server.py   (API :8766, widget at /)

Today (IST) is injected into the system prompt so "tomorrow" resolves correctly.
"""
import base64
import io
import json
import os
import subprocess
import tempfile
import time
import wave
from datetime import datetime
from zoneinfo import ZoneInfo

os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from brain import agentic_turn
from tools import run_tool

BASE = os.path.dirname(os.path.abspath(__file__))
VOICE_BASE = "/home/hatch/workspace/pixalified-voice-agent"
PIPER_MODEL = os.path.join(VOICE_BASE, "models", "piper", "en_US-amy-medium.onnx")
WHISPER_MODEL = os.path.join(VOICE_BASE, "models", "whisper-tiny")
IST = ZoneInfo("Asia/Kolkata")

app = FastAPI(title="Receptionist v2 (agentic, local tools)")

_whisper = None
_piper = None


def get_whisper():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    return _whisper


def get_piper():
    global _piper
    if _piper is None:
        from piper import PiperVoice
        _piper = PiperVoice.load(PIPER_MODEL)
    return _piper


def convert_to_wav16k(raw: bytes) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".in", delete=False) as f:
        f.write(raw)
        in_path = f.name
    out_path = in_path + ".wav"
    try:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", in_path,
                        "-ar", "16000", "-ac", "1", "-f", "wav", out_path],
                       check=True, timeout=30)
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        for p in (in_path, out_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def transcribe(wav_bytes: bytes) -> str:
    model = get_whisper()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        path = f.name
    try:
        segments, _ = model.transcribe(path, beam_size=1, language="en")
        return " ".join(s.text.strip() for s in segments).strip()
    finally:
        os.unlink(path)


def synthesize(text: str) -> bytes:
    voice = get_piper()
    chunks = [(c.audio_float_array * 32767).astype(np.int16) for c in voice.synthesize(text)]
    pcm = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(voice.config.sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def do_turn(transcript: str) -> dict:
    today_str = datetime.now(IST).strftime("%Y-%m-%d %A")
    turn = agentic_turn(transcript, today_str)
    t0 = time.time()
    reply_wav = synthesize(turn["spoken_text"])
    return {
        "transcript": transcript,
        "reply_text": turn["reply_text"],
        "tool_calls": turn["tool_calls"],
        "audio_b64": base64.b64encode(reply_wav).decode(),
        "tts_ms": round((time.time() - t0) * 1000),
    }


class TextTurn(BaseModel):
    text: str


@app.post("/api/text-turn")
def text_turn(body: TextTurn):
    """Scripted turn: text in -> brain + tools -> TTS audio out (no mic needed)."""
    return JSONResponse(do_turn(body.text))


@app.post("/api/turn")
async def voice_turn(audio: UploadFile = File(...)):
    t_all = time.time()
    raw = await audio.read()
    wav = convert_to_wav16k(raw)
    transcript = transcribe(wav) or "(no speech detected)"
    d = do_turn(transcript)
    d["latency_total_ms"] = round((time.time() - t_all) * 1000)
    return JSONResponse(d)


@app.get("/api/health")
def health():
    return {"ok": True, "whisper": os.path.isdir(WHISPER_MODEL),
            "piper": os.path.exists(PIPER_MODEL)}


@app.get("/")
def widget():
    return FileResponse(os.path.join(BASE, "widget.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8766)
