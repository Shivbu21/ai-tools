# Hinglish Speech-to-Text (optimized decoding)

Real, working Hindi-English code-switched transcription on CPU using
**faster-whisper** (tiny, int8). The trick: force `language="hi"` and pass an
`initial_prompt` of code-switched example sentences — Whisper's decoder
conditions on the prompt, which steers it to keep *both* Hindi and English
spans instead of collapsing everything into one language/script.

## One-command run

```bash
~/workspace/pixalified-voice-agent/venv/bin/python transcribe.py samples/clip1.wav
```

Other modes:

```bash
# Default decoding (auto language, no prompt) — the baseline we compare against
~/workspace/pixalified-voice-agent/venv/bin/python transcribe.py samples/clip1.wav --language auto --no-prompt

# JSON output with word-level segments + timestamps
~/workspace/pixalified-voice-agent/venv/bin/python transcribe.py audio.mp3 --json

# Reproduce the eval table below
~/workspace/pixalified-voice-agent/venv/bin/python eval.py
```

## Measured results (real runs, 2026-10-14)

3 synthetic Hinglish clips (Piper `hi_IN-priyamvada-medium` TTS), WER against
ground truth after Devanagari→Latin normalization (see `normalize.py`):

| clip | baseline WER (auto lang, no prompt) | optimized WER (lang=hi + prompt) |
|---|---|---|
| clip1.wav | 0.615 | 0.308 |
| clip2.wav | 0.818 | 0.636 |
| clip3.wav | 0.583 | **0.000** |
| **AVERAGE** | **0.672** | **0.315** |

**Absolute WER change: +0.358 (+53.2% relative improvement vs baseline).**
Example — clip3 baseline: *"This phone's pair skyo high and does it come with
uvaranti."* → optimized: *"yeh phone ka price kya hai and does it come with
warranty?"* (exact match).

Full per-clip hypotheses: `results/eval_results.json`.

## Honest limitations

- **Absolute WER is still high** (~0.32 avg). whisper-tiny is a 39M-parameter
  model; on fast synthetic Hinglish it mishears words ("ten o'clock" →
  "atten"). The *relative* gain from the prompt is the real, reproducible
  signal here — not production-grade accuracy.
- **Synthetic eval set.** The 3 clips are Piper TTS, not natural human
  code-switching (documented in `generate_samples.py`). Real-world accuracy
  will differ; the fine-tune pipeline exists to close that gap with real
  Common Voice Hindi speech.
- **Normalization caveat.** WER is computed after transliterating Devanagari
  to Latin and lowercasing, so pure script choice isn't penalized. Without
  this, scores would look worse for reasons that don't reflect understanding.
- **Fine-tuning needs a GPU.** `finetune.py` refuses to train on CPU (use
  `--smoke-test` to validate the wiring — verified passing 2026-10-14: data
  load → collator → forward+backward OK, loss 5.227 on 1 step).
  `prep_data.py` builds the manifest from local clips + an optional Common
  Voice Hindi slice (gated HF dataset — needs `huggingface-cli login`).
  Sandbox note: unset the bracketed-IPv6 `no_proxy` entries
  (`env -u no_proxy -u NO_PROXY`) when downloading from the HF Hub —
  huggingface_hub's vendored httpx fails to parse `[::1]`.
- Hindi Piper voice note: `hi_IN-pratham-medium` consistently downloaded
  corrupt (protobuf parse failure, tried twice); `hi_IN-priyamvada-medium`
  works and is bundled under `models/piper/`.

## Files

| file | what |
|---|---|
| `transcribe.py` | CLI: audio → transcript + timestamped segments |
| `eval.py` | before/after eval, prints table, writes `results/` |
| `generate_samples.py` | synthesizes `samples/*.wav` + `transcripts.json` via Piper |
| `normalize.py` | transliteration normalization + WER implementation |
| `prep_data.py` | fine-tune manifest (local clips + Common Voice Hindi slice) |
| `finetune.py` | Whisper Seq2SeqTrainer fine-tune (**GPU required**) |
| `eval_finetune.py` | WER of a fine-tuned checkpoint on the dev set |
