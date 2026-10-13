# Avatar Lip-Sync (Wav2Lip, CPU)

True lip-sync talking head: give it a portrait photo and an audio file, get
back an mp4 of the face speaking the audio with synced lip movements. Real
Wav2Lip GAN inference (`wav2lip_gan.pth`), running on CPU — slow, but genuine
model output, no faked animation.

Built as the engine for the user's Instagram AI-avatar pipeline
(`@pixaalified` female AI spokesperson).

## One-command run

```bash
# First time only (~15-20 min: torch CPU, 415MB GAN weights, face-detector weights)
bash setup.sh

# Lip-sync (uses ./venv created by setup.sh)
./venv/bin/python lipsync.py --face assets/portrait.png --audio assets/voice.wav --out results/demo.mp4

# Verify the output (streams, duration, lips actually move)
./venv/bin/python verify.py --out results/demo.mp4 --face assets/portrait.png
```

## Measured run (real, 2026-10-14)

- Input: AI-generated synthetic portrait `assets/portrait.png` (1600×1600,
  clearly labeled synthetic — **not a real person's photo**) + 9.75s Hindi/
  English voice line `assets/voice.wav` (Piper `hi_IN-priyamvada-medium`:
  *"Namaste! Mera naam Maya hai. Main Pixalified ki AI assistant hoon, and I
  build beautiful websites for local businesses."*)
- Output: `results/demo.mp4` — h264 1600×1600 @ 25fps, **video + audio
  streams, 9.75s**
- Runtime: **161s for 9.75s of video on 2 vCPUs (CPU-only)** ≈ 16.5×
  slower than realtime
- `verify.py` result: **PASS** — mouth region (located from the SFD face
  box) mean abs diff between sampled frames: **3.34 / 10.58 / 9.68**;
  vs the static input portrait: **7.90–13.25**. The lips visibly open/close
  across frames (checked visually frame-by-frame).

## Honest limitations

- **Slow on CPU.** ~16× realtime on 2 vCPUs. A 60s Instagram reel would take
  ~16 min. A GPU (even a modest one) cuts this to near-realtime. Batch sizes
  in `lipsync.py` default to 8/4 for a ~2.5GB RAM budget — raise them on
  bigger machines.
- **Quality caveats.** Wav2Lip regenerates only the cropped face region at
  96×96 and pastes it back; on close inspection there is softness around the
  mouth and slight identity drift vs the portrait. Good enough for a
  stylized avatar reel, not for photoreal close-ups.
- **Portrait is synthetic.** `assets/portrait.png` was AI-generated for this
  project and is labeled as such. Swap in any front-facing portrait —
  face detection needs a clear, front-facing face; profile/occluded faces
  fail with "no face detected".
- **Setup downloads ~700MB** (torch CPU ~200MB, GAN weights 415MB, SFD face
  weights 90MB). The repo's own weight links were unreliable, so `setup.sh`
  pulls `wav2lip_gan.pth` from a HuggingFace community mirror and applies
  three small compat patches for modern numpy/torch/librosa
  (`face_detection/utils.py` `np.int`, `sfd_detector.py` `map_location`,
  `audio.py` keyword-only `mel()`).
- Sandbox note: `setup.sh` unsets the bracketed-IPv6 `no_proxy` entries when
  calling huggingface_hub (its vendored httpx fails to parse `[::1]`).

## Files

| file | what |
|---|---|
| `setup.sh` | venv + CPU torch + Wav2Lip clone + weights + patches (idempotent) |
| `lipsync.py` | CLI: `--face --audio --out` → runs `Wav2Lip/inference.py` on CPU |
| `verify.py` | ffprobe stream/duration checks + mouth-region motion check via SFD face box |
| `assets/portrait.png` | synthetic AI-generated portrait (labeled, not a real person) |
| `assets/voice.wav` | 9.75s Piper Hindi-voice line used for the demo |
| `results/demo.mp4` | verified lip-synced output |
| `checkpoints/` | `wav2lip_gan.pth` (415MB, downloaded by setup.sh) |
