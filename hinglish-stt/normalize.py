"""Shared text normalization + WER for Hinglish STT evaluation.

Whisper with language="hi" renders Hindi words in Devanagari script, while our
ground-truth transcripts are written in Latin-script Hinglish (the way people
actually type it). Comparing raw strings would punish correct transcriptions
purely for script choice, so we transliterate Devanagari -> Latin (ITRANS),
lowercase, strip punctuation, and collapse whitespace before scoring. The
normalization is applied identically to reference and hypothesis, and is
documented in the README as part of the eval protocol.
"""

import re
import unicodedata

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate

    _HAS_TRANSLIT = True
except ImportError:  # pragma: no cover
    _HAS_TRANSLIT = False

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize a transcript for fair WER comparison."""
    text = unicodedata.normalize("NFC", text)
    if _HAS_TRANSLIT:
        # Leaves ASCII/Latin characters untouched, converts Devanagari spans.
        text = transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
    text = text.lower()
    text = _PUNCT_RE.sub("", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate = (S + D + I) / N, standard DP implementation."""
    ref = reference.split()
    hyp = hypothesis.split()
    if not ref:
        return 0.0 if not hyp else 1.0
    # d[i][j]: edit distance between ref[:i] and hyp[:j]
    prev = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        cur = [i] + [0] * len(hyp)
        ri = ref[i - 1]
        for j in range(1, len(hyp) + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[len(hyp)] / len(ref)


def wer_report(reference: str, hypothesis: str) -> dict:
    """WER plus the normalized strings, for debugging/inspection."""
    ref_n = normalize_text(reference)
    hyp_n = normalize_text(hypothesis)
    return {
        "wer": wer(ref_n, hyp_n),
        "ref_norm": ref_n,
        "hyp_norm": hyp_n,
    }
