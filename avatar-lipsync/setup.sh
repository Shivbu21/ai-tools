#!/usr/bin/env bash
# Sets up Wav2Lip for CPU inference. Idempotent — safe to re-run.
#   bash setup.sh
# Time: ~10-20 min first run (torch CPU ~200MB, wav2lip_gan.pth ~436MB).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TMPDIR="${TMPDIR:-$HOME/workspace/.tmp}"   # keep pip off the 512M /tmp tmpfs
mkdir -p "$TMPDIR"

echo "==> [1/6] Python venv"
if [ ! -d "$ROOT/venv" ]; then python3 -m venv "$ROOT/venv"; fi
PY="$ROOT/venv/bin/python"
"$ROOT/venv/bin/pip" install -q -U pip

echo "==> [2/6] torch + torchvision (CPU builds)"
"$ROOT/venv/bin/pip" install -q torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

echo "==> [3/6] Wav2Lip source (github.com/Rudrabha/Wav2Lip)"
if [ ! -d "$ROOT/Wav2Lip" ]; then
    git clone -q https://github.com/Rudrabha/Wav2Lip "$ROOT/Wav2Lip"
fi

echo "==> [4/6] Python deps (modern equivalents of requirements.txt)"
"$ROOT/venv/bin/pip" install -q \
    numpy scipy librosa opencv-python-headless tqdm numba \
    huggingface_hub soundfile

echo "==> [5/6] Compatibility patches for modern numpy/torch/librosa"
# face_detection/utils.py uses np.int (removed in numpy>=1.24); the import
# chain `import face_detection` -> api -> utils hits it on every run.
sed -i 's/dtype=np\.int)/dtype=int)/' "$ROOT/Wav2Lip/face_detection/utils.py"
# sfd_detector torch.load without map_location breaks on CPU-only boxes if
# weights were cuda-saved; route through the detector's own device string.
"$PY" - <<'EOF'
import os, re
root = os.path.expanduser("~/workspace/ai-tools/avatar-lipsync")
p = os.path.join(root, "Wav2Lip/face_detection/detection/sfd/sfd_detector.py")
src = open(p).read()
src = src.replace("model_weights = torch.load(path_to_detector)",
                  "model_weights = torch.load(path_to_detector, map_location=device)")
open(p, "w").write(src)

# audio.save_wavenet_wav uses librosa.output (removed in librosa>=0.8);
# unused by inference, but keep the module import-safe for future callers.
p2 = os.path.join(root, "Wav2Lip/audio.py")
src = open(p2).read()
src = src.replace("librosa.output.write_wav(path, wav, sr=sr)",
                  "wavfile.write(path, sr, (wav * 32767).astype(np.int16))")
# librosa>=0.10 made filters.mel() keyword-only; Wav2Lip passes sr/n_fft
# positionally.
src = src.replace(
    "librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels,",
    "librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, "
    "n_mels=hp.num_mels,")
open(p2, "w").write(src)
print("patches applied")
EOF

echo "==> [6/6] Model weights"
mkdir -p "$ROOT/checkpoints"
# NOTE: the sandbox's no_proxy contains bracketed IPv6 entries ([::1]) that
# huggingface_hub's vendored httpx fails to parse -> unset for this step.
env -u no_proxy -u NO_PROXY ROOT="$ROOT" "$PY" - <<'EOF'
import os, shutil
from huggingface_hub import hf_hub_download
root = os.environ["ROOT"]
dst = os.path.join(root, "checkpoints", "wav2lip_gan.pth")
if not os.path.isfile(dst):
    # Community mirror of the official Wav2Lip-GAN weights (repo's own
    # SharePoint link is unreliable); 436MB.
    p = hf_hub_download(repo_id="Non-playing-Character/Wave2lip",
                        filename="wav2lip_gan.pth")
    shutil.copy(p, dst)
print("weights OK:", dst, os.path.getsize(dst) // 1024 // 1024, "MB")
EOF

# Face-detection (SFD) weights: the bundled detector auto-downloads these on
# first run; pre-fetch to the expected path so setup is the only network step.
SFD="$ROOT/Wav2Lip/face_detection/detection/sfd/s3fd.pth"
if [ ! -f "$SFD" ]; then
    curl -sSL -o "$SFD" https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth
fi
echo "face-detection weights OK"

echo
echo "Setup complete. Try:"
echo "  $PY lipsync.py --face assets/portrait.png --audio assets/voice.wav --out results/demo.mp4"
