"""
forensics.py — real, implemented forensic indicators for payment screenshots.

NOT a fraud detector. Every function below implements an actual measurement;
the report strings say what was measured and what it can/can't mean.

Checks:
  1. Metadata (EXIF / PNG text chunks / JPEG headers) via PIL — presence or absence.
  2. Error Level Analysis (ELA): re-save at JPEG quality 90, absolute difference
     of the luminance channel, amplified -> heatmap PNG. Regions whose mean error
     is far above the image's own baseline get flagged as possible edits.
  3. Layout/template heuristics for GPay / PhonePe / Paytm receipts: expected
     aspect ratio bands, header band brightness, a centred large-amount region.
     These are structural heuristics ONLY — fonts/glyphs are not checked.
  4. Risk score 0-100 = weighted sum of the indicators above, every contribution
     listed and explained. No black box.

Honesty rule: metadata absence is a *mild* flag (screenshots are often stripped
by chat apps), a clean ELA is NOT a clean bill of health, and none of this
proves tampering or authenticity.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ExifTags


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Indicator:
    name: str
    weight: float          # weight in the 0-100 risk score
    value: float           # measured value, 0..1 risk contribution scale
    points: float          # weight * value (what actually went into the score)
    measured: str          # what was actually measured
    meaning: str           # plain-language interpretation + caveat


@dataclass
class Report:
    width: int
    height: int
    fmt: str | None
    mode: str
    indicators: list[Indicator] = field(default_factory=list)
    risk_score: float = 0.0
    ela_heatmap: Image.Image | None = None
    ela_flag_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Image: {self.width}x{self.height} {self.fmt} ({self.mode})",
            f"RISK SCORE: {self.risk_score:.1f} / 100",
            "",
            "Indicators (each shows its exact contribution):",
        ]
        for i in self.indicators:
            lines.append(
                f"  [{i.name}] +{i.points:.1f} pts "
                f"(weight {i.weight}, risk {i.value:.2f})\n"
                f"    measured: {i.measured}\n"
                f"    meaning:  {i.meaning}"
            )
        if self.notes:
            lines.append("\nNotes:")
            lines += [f"  - {n}" for n in self.notes]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. Metadata
# ---------------------------------------------------------------------------

def extract_metadata(img: Image.Image) -> dict:
    """Extract whatever metadata PIL can see. Returns a flat dict of findings."""
    meta: dict = {}
    meta["format"] = img.format
    meta["mode"] = img.mode

    # EXIF (JPEG / TIFF)
    try:
        exif = img.getexif()
        if exif:
            for tag_id, value in exif.items():
                tag = ExifTags.TAGS.get(tag_id, str(tag_id))
                meta[f"exif:{tag}"] = str(value)
    except Exception as e:  # pragma: no cover - defensive
        meta["exif_error"] = str(e)

    # PNG text chunks / other info dict
    for k, v in (img.info or {}).items():
        if k == "exif":
            continue
        try:
            meta[f"pnginfo:{k}"] = str(v)[:200]
        except Exception:
            pass

    # JPEG quantization tables are a genuine forensic signal: many phones and
    # chat apps re-save with their own tables. PIL exposes them on JPEGs.
    if img.format == "JPEG":
        meta["jpeg_has_qtables"] = "qtables" in (img.info or {})
    return meta


def check_metadata(img: Image.Image) -> Indicator:
    meta = extract_metadata(img)
    interesting = {k: v for k, v in meta.items()
                   if k.startswith(("exif:", "pnginfo:")) or k == "jpeg_has_qtables"}

    has_device_info = any(
        k.split(":")[-1] in ("Make", "Model", "Software", "CreateDate",
                             "DateTimeOriginal", "HostComputer")
        for k in interesting
    )
    # Screenshots forwarded through chat apps usually lose all metadata, so
    # absence is only a MILD signal, never a verdict. Presence of device info
    # is mildly reassuring but proves nothing (EXIF is trivial to write).
    if has_device_info:
        risk, measured = 0.05, f"{len(interesting)} metadata fields incl. device info"
        meaning = ("Device metadata present. Mildly consistent with an original "
                   "screenshot; EXIF is easy to forge, so this is not proof of authenticity.")
    elif interesting:
        risk, measured = 0.25, f"{len(interesting)} metadata fields, no device info"
        meaning = ("Some metadata present but nothing device-specific. Neutral; "
                   "chat apps commonly strip metadata when forwarding.")
    else:
        risk, measured = 0.35, "no EXIF, PNG info chunks, or JPEG qtables found"
        meaning = ("Metadata absent. Mild flag only: WhatsApp/Telegram strip metadata "
                   "on forward, and 'share as document' vs 'as image' changes this. "
                   "Not a verdict of tampering.")

    return Indicator(
        name="metadata",
        weight=15.0,
        value=risk,
        points=15.0 * risk,
        measured=measured,
        meaning=meaning,
    )


# ---------------------------------------------------------------------------
# 2. Error Level Analysis
# ---------------------------------------------------------------------------

# Standard IJG luminance quantization table (quality-50 base) for estimating
# the JPEG quality a file was saved at — the estimate drives the
# self-consistency ELA pass.
_STD_LUMA_QTABLE = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]


def estimate_jpeg_quality(img: Image.Image) -> int | None:
    """Estimate the JPEG save quality from the luminance quantization table.

    Returns None for non-JPEGs / missing tables. Pure measurement: finds the
    standard IJG quality whose scaled tables best match the file's tables.
    """
    qtables = getattr(img, "quantization", None)
    if not qtables:
        return None
    qtable = list(qtables[0])[:64]
    if len(qtable) < 64:
        return None
    best, best_err = None, float("inf")
    for q in range(5, 101):
        s = 5000 / q if q < 50 else 200 - 2 * q
        pred = [min(255, max(1, (v * s + 50) // 100)) for v in _STD_LUMA_QTABLE]
        err = sum((a - b) ** 2 for a, b in zip(qtable, pred))
        if err < best_err:
            best, best_err = q, err
    return best


def _luma_diff(img: Image.Image, quality: int) -> np.ndarray:
    """Absolute luma difference between img and its JPEG re-save at `quality`."""
    rgb = img.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")
    a = np.asarray(rgb, dtype=np.int16)
    b = np.asarray(resaved, dtype=np.int16)
    return np.abs(0.299 * (a[..., 0] - b[..., 0])
                  + 0.587 * (a[..., 1] - b[..., 1])
                  + 0.114 * (a[..., 2] - b[..., 2]))


def ela_heatmap(img: Image.Image, quality: int = 90) -> Image.Image:
    """ELA heatmap PNG (black -> red -> yellow), percentile-clipped for display."""
    diff = _luma_diff(img, quality)
    vmax = float(np.percentile(diff, 99.5))
    norm = np.clip(diff / max(vmax, 1e-6), 0, 1)
    heat = np.zeros((diff.shape[0], diff.shape[1], 3), dtype=np.uint8)
    heat[..., 0] = (255 * np.clip(norm * 2, 0, 1)).astype(np.uint8)
    heat[..., 1] = (255 * np.clip(norm * 2 - 1, 0, 1)).astype(np.uint8)
    return Image.fromarray(heat)


def _cell_means(diff: np.ndarray, grid: int) -> np.ndarray:
    h, w = diff.shape
    gh, gw = (h + grid - 1) // grid, (w + grid - 1) // grid
    cms = np.zeros((gh, gw))
    for gy in range(gh):
        for gx in range(gw):
            cell = diff[gy * grid:(gy + 1) * grid, gx * grid:(gx + 1) * grid]
            cms[gy, gx] = cell.mean() if cell.size else 0.0
    return cms


def _components(mask: np.ndarray, grid: int, img_w: int, img_h: int):
    """4-connected components of a boolean cell mask -> dicts with bbox."""
    gh, gw = mask.shape
    seen = np.zeros_like(mask, bool)
    comps = []
    for gy in range(gh):
        for gx in range(gw):
            if mask[gy, gx] and not seen[gy, gx]:
                stack = [(gy, gx)]
                seen[gy, gx] = True
                cells = []
                while stack:
                    cy, cx = stack.pop()
                    cells.append((cy, cx))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if (0 <= ny < gh and 0 <= nx < gw
                                and mask[ny, nx] and not seen[ny, nx]):
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                ys = [c[0] for c in cells]
                xs = [c[1] for c in cells]
                comps.append({
                    "cells": cells,
                    "n": len(cells),
                    "bbox": (min(xs) * grid, min(ys) * grid,
                             min(max(xs) * grid + grid, img_w),
                             min(max(ys) * grid + grid, img_h)),
                })
    return comps


def find_suspicious_regions(img: Image.Image, grid: int = 24):
    """Flag large, coherently-hot regions whose error does NOT stabilise.

    Method (all measured, no magic):
      1. ELA@90 -> 24px cell means. Seed mask: cell > max(2 x p75, 0.5).
         (p75 = robust "normal text" level; median would be flat background.)
      2. 4-connected components of seeded cells.
      3. A component is suspicious iff it is LARGE (n >= 20 cells — a forged
         amount/region is a fat blob, natural text strokes are thin) AND hot
         (mean cell error > 2.3) AND — when the JPEG save quality Q can be
         estimated — it does not stabilise under ELA@Q (mean cell error there
         > 0.06). Genuine same-pipeline content goes near-zero at ELA@Q;
         composited content from a differently-compressed source does not.

    Thresholds were calibrated on the synthetic test images in samples/ and
    are reported as such — they are not universal forgery constants.

    Returns (boxes, diagnostics_dict).
    """
    w, h = img.size
    diff90 = _luma_diff(img, 90)
    cms90 = _cell_means(diff90, grid)
    p75 = float(np.percentile(cms90, 75))
    seed_thr = max(2.0 * p75, 0.5)

    q_est = estimate_jpeg_quality(img)
    cms_q = _cell_means(_luma_diff(img, q_est), grid) if q_est else None

    boxes = []
    diag = {"q_est": q_est, "p75": round(p75, 3), "seed_thr": round(seed_thr, 3),
            "components": []}
    for comp in _components(cms90 > seed_thr, grid, w, h):
        vals90 = [cms90[cy, cx] for cy, cx in comp["cells"]]
        mean90 = float(np.mean(vals90))
        mean_q = (float(np.mean([cms_q[cy, cx] for cy, cx in comp["cells"]]))
                  if cms_q is not None else None)
        suspicious = (comp["n"] >= 20 and mean90 > 2.3
                      and (mean_q is None or mean_q > 0.06))
        diag["components"].append(
            {"n": comp["n"], "mean90": round(mean90, 2),
             "mean_q": round(mean_q, 3) if mean_q is not None else None,
             "bbox": comp["bbox"], "suspicious": suspicious})
        if suspicious:
            boxes.append(comp["bbox"])
    diag["components"].sort(key=lambda c: -c["mean90"])
    return boxes, diag


def check_ela(img: Image.Image) -> tuple[Indicator, Image.Image,
                                        list[tuple[int, int, int, int]]]:
    heat = ela_heatmap(img, 90)
    boxes, diag = find_suspicious_regions(img)

    annotated = heat.copy()
    d = ImageDraw.Draw(annotated)
    for (x0, y0, x1, y1) in boxes:
        d.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=3)

    q_note = (f"estimated JPEG quality {diag['q_est']}"
              if diag["q_est"] else "quality not estimable (not JPEG?)")
    if not boxes:
        risk = 0.10
        measured = (f"ELA@90, {q_note}; no large coherent hot region "
                    f"(seed>{diag['seed_thr']}, need >=20 connected cells "
                    f"with mean cell error >2.3" +
                    (f" and ELA@{diag['q_est']} mean >0.06" if diag["q_est"] else "") +
                    ")")
        meaning = ("No region stands out vs the image baseline. Does NOT prove the "
                   "image is genuine — a forgery edited and re-saved in one pipeline "
                   "is nearly invisible to ELA. Absence of flags is weak evidence.")
    else:
        risk = 0.85
        sus = [c for c in diag["components"] if c["suspicious"]]
        measured = (f"ELA@90, {q_note}; {len(boxes)} suspicious region(s): {boxes}; "
                    f"component detail: {sus}")
        meaning = (f"{len(boxes)} large region(s) are hot under ELA@90 yet do not "
                   f"stabilise at the image's own estimated quality — consistent with "
                   "pixels composited from a differently-compressed source (a common "
                   "forgery pattern). Possible tampering: verify the underlying "
                   "transaction in your own UPI app / bank statement. Indicator, not proof.")

    ind = Indicator(name="ela", weight=40.0, value=risk, points=40.0 * risk,
                    measured=measured, meaning=meaning)
    return ind, annotated, boxes


# ---------------------------------------------------------------------------
# 3. Layout / template heuristics (structural only)
# ---------------------------------------------------------------------------

KNOWN_FORMATS = {
    "gpay":    {"aspect": (0.45, 0.55), "header_frac": 0.18,
                "desc": "Google Pay receipt: tall narrow (~9:19.5), dark header band"},
    "phonepe": {"aspect": (0.45, 0.55), "header_frac": 0.22,
                "desc": "PhonePe receipt: tall narrow, purple header band"},
    "paytm":   {"aspect": (0.48, 0.60), "header_frac": 0.20,
                "desc": "Paytm receipt: tall narrow, blue header band"},
}


def _band_brightness(gray: np.ndarray, y0: float, y1: float) -> float:
    h = gray.shape[0]
    band = gray[int(h * y0):int(h * y1)]
    return float(band.mean()) if band.size else 128.0


def check_layout(img: Image.Image) -> Indicator:
    rgb = img.convert("RGB")
    w, h = rgb.size
    aspect = w / h
    gray = np.asarray(rgb.convert("L"), dtype=np.float32)

    # which known format is the aspect closest to?
    best, best_spec, best_gap = None, None, 1e9
    for name, spec in KNOWN_FORMATS.items():
        lo, hi = spec["aspect"]
        mid = (lo + hi) / 2
        gap = abs(aspect - mid)
        if gap < best_gap:
            best, best_spec, best_gap = name, spec, gap
    mid = (best_spec["aspect"][0] + best_spec["aspect"][1]) / 2
    in_band = best_spec["aspect"][0] <= aspect <= best_spec["aspect"][1]

    # header band vs body brightness contrast (receipts have a distinct header)
    hf = best_spec["header_frac"]
    header_b = _band_brightness(gray, 0.0, hf)
    body_b = _band_brightness(gray, hf, 0.9)
    contrast = abs(header_b - body_b)

    # centred large-amount region: receipts show the amount big near the top
    cx0, cx1 = int(w * 0.2), int(w * 0.8)
    amt_band = gray[int(h * 0.20):int(h * 0.45), cx0:cx1]
    amt_contrast = float(amt_band.std()) if amt_band.size else 0.0

    checks_passed = 0
    detail = []
    if in_band:
        checks_passed += 1
        detail.append(f"aspect {aspect:.2f} inside {best} band {best_spec['aspect']}")
    else:
        detail.append(f"aspect {aspect:.2f} OUTSIDE {best} band {best_spec['aspect']}")
    if contrast > 15:
        checks_passed += 1
        detail.append(f"header/body contrast {contrast:.0f} (distinct header)")
    else:
        detail.append(f"header/body contrast {contrast:.0f} (flat, unusual for receipts)")
    if amt_contrast > 25:
        checks_passed += 1
        detail.append(f"amount-zone texture {amt_contrast:.0f} (busy, like big text)")
    else:
        detail.append(f"amount-zone texture {amt_contrast:.0f} (flat, no bold amount?)")

    risk = {3: 0.05, 2: 0.35, 1: 0.65, 0: 0.9}[checks_passed]
    measured = f"closest template '{best}'; " + "; ".join(detail)
    if checks_passed == 3:
        meaning = ("Structure resembles a known UPI receipt format. This only means "
                   "the layout was copied well — a forgery made from the real app "
                   "would pass this too.")
    else:
        meaning = ("Structure deviates from known UPI receipt layouts. Could be a "
                   "different app/version (we only model 3), a cropped screenshot, "
                   "or a fabricated receipt. Treat as a reason to look closer, not "
                   "a verdict. Fonts, logos and glyphs are NOT checked.")

    return Indicator(name="layout", weight=20.0, value=risk, points=20.0 * risk,
                     measured=measured, meaning=meaning)


# ---------------------------------------------------------------------------
# 4. Extra cheap signal: double-JPEG / resave hints via qtable count
# ---------------------------------------------------------------------------

def check_resave_hints(img: Image.Image) -> Indicator:
    w, h = img.size
    # Screenshots pasted into editors and re-saved often carry tell-tale size
    # artefacts; the honest cheap version: report whether the image dimensions
    # look like a native phone screenshot vs something resized.
    common = [(1080, 2400), (1080, 2340), (720, 1600), (1080, 1920),
              (1440, 3200), (1170, 2532), (1284, 2778)]
    native = any(abs(w - cw) < 40 and abs(h - ch) < 40 for cw, ch in common)
    if native:
        risk, measured = 0.1, f"{w}x{h} matches a common phone screenshot size"
        meaning = ("Dimensions look like a native screenshot. Reassuring but weak: "
                   "forgeries are often made at native resolution too.")
    else:
        risk, measured = 0.4, f"{w}x{h} does not match common phone screenshot sizes"
        meaning = ("Unusual dimensions — could be cropped, resized, or from an "
                   "uncommon device. Mild flag; cropping alone is not tampering.")
    return Indicator(name="dimensions", weight=10.0, value=risk,
                     points=10.0 * risk, measured=measured, meaning=meaning)


# size sanity: absurd file sizes for the dimensions can hint at re-encoding
def check_file_size(path: str, img: Image.Image) -> Indicator:
    import os
    nbytes = os.path.getsize(path)
    px = img.size[0] * img.size[1]
    bpp = nbytes * 8 / max(px, 1)
    if bpp > 6:
        risk, measured = 0.3, f"{nbytes/1024:.0f} KB for {px/1e6:.1f} MP ({bpp:.1f} bits/px — high)"
        meaning = ("Unusually large file for these dimensions can mean a PNG "
                   "re-save of a JPEG (quality laundering). Mild flag only.")
    elif bpp < 0.15:
        risk, measured = 0.25, f"{nbytes/1024:.0f} KB for {px/1e6:.1f} MP ({bpp:.2f} bits/px — very low)"
        meaning = ("Very heavy compression can hide edit artefacts from ELA. "
                   "Low score confidence either way.")
    else:
        risk, measured = 0.05, f"{nbytes/1024:.0f} KB for {px/1e6:.1f} MP ({bpp:.2f} bits/px — normal)"
        meaning = "File size is ordinary for these dimensions."
    return Indicator(name="filesize", weight=5.0, value=risk,
                     points=5.0 * risk, measured=measured, meaning=meaning)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

WEIGHTS_TOTAL = 15.0 + 40.0 + 20.0 + 10.0 + 5.0  # = 90 -> normalised to 100


def analyze(path: str) -> Report:
    img = Image.open(path)
    rep = Report(width=img.size[0], height=img.size[1],
                 fmt=img.format, mode=img.mode)

    rep.indicators.append(check_metadata(img))
    ela_ind, heat, boxes = check_ela(img)
    rep.indicators.append(ela_ind)
    rep.ela_heatmap = heat
    rep.ela_flag_boxes = boxes
    rep.indicators.append(check_layout(img))
    rep.indicators.append(check_resave_hints(img))
    rep.indicators.append(check_file_size(path, img))

    raw = sum(i.points for i in rep.indicators)
    rep.risk_score = round(raw * 100.0 / WEIGHTS_TOTAL, 1)

    rep.notes = [
        "These are indicators of possible tampering, NOT proof.",
        "A low score does NOT guarantee authenticity — a careful forgery can score low.",
        "Never use this as the sole basis for financial decisions; always verify the "
        "transaction in your own UPI app / bank statement.",
    ]
    return rep


if __name__ == "__main__":
    import sys
    r = analyze(sys.argv[1])
    print(r.summary())
    if r.ela_heatmap and len(sys.argv) > 2:
        r.ela_heatmap.save(sys.argv[2])
        print("heatmap ->", sys.argv[2])
