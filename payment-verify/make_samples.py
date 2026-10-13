"""Generate test assets:
  samples/clean_receipt.jpg    - synthetic receipt-like image (JPEG q92, with EXIF)
  samples/tampered_receipt.jpg - SAME image, but the amount digits are REALLY
                                 re-painted with PIL (changed 2,450 -> 9,850)
                                 then re-saved. No mock: actual pixel edits.
Run: python3 make_samples.py
"""
from PIL import Image, ImageDraw, ImageFont, ImageOps
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
os.makedirs(OUT, exist_ok=True)

W, H = 540, 1170  # phone-like portrait
BG = (245, 247, 250)
HEADER = (31, 41, 55)
GREEN = (22, 163, 74)
INK = (17, 24, 39)
MUTED = (107, 114, 128)
WHITE = (255, 255, 255)


def font(sz):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", sz)
    except Exception:
        return ImageFont.load_default(size=sz)


def draw_receipt(amount_text):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    # header band
    d.rectangle([0, 0, W, 200], fill=HEADER)
    d.text((W // 2, 80), "PaySample", font=font(44), fill=WHITE, anchor="mm")
    d.text((W // 2, 135), "UPI Receipt", font=font(28), fill=(180, 190, 205), anchor="mm")
    # success check + label
    d.ellipse([W // 2 - 45, 245, W // 2 + 45, 335], fill=GREEN)
    d.text((W // 2, 292), "\u2713", font=font(52), fill=WHITE, anchor="mm")
    d.text((W // 2, 380), "Payment successful", font=font(34), fill=INK, anchor="mm")
    # amount — big, centred (this is the region we will tamper)
    d.text((W // 2, 480), amount_text, font=font(72), fill=INK, anchor="mm")
    d.text((W // 2, 545), "Paid to Sharma General Store", font=font(28), fill=MUTED, anchor="mm")
    # details card
    d.rounded_rectangle([40, 620, W - 40, 980], radius=24, fill=WHITE)
    rows = [
        ("UPI Ref No", "418229736105"),
        ("Date", "13 Oct 2026, 09:41 PM"),
        ("From", "shivam •••• 4108"),
        ("UPI ID", "shivam@okhdfc"),
        ("Status", "SUCCESS"),
    ]
    y = 670
    for k, v in rows:
        d.text((70, y), k, font=font(24), fill=MUTED, anchor="lm")
        d.text((W - 70, y), v, font=font(24), fill=INK, anchor="rm")
        y += 58
    d.text((W // 2, 1060), "SAMPLE IMAGE - generated for testing",
           font=font(20), fill=MUTED, anchor="mm")
    # mild texture so ELA has a realistic baseline
    img = ImageOps.exif_transpose(img)
    return img


def save_jpeg(img, path, exif_dict=None, quality=92):
    ex = Image.Exif()
    for k, v in (exif_dict or {}).items():
        ex[k] = v
    img.save(path, "JPEG", quality=quality, exif=ex)


# 1) clean — single-generation JPEG q95, like an original high-quality screenshot
clean = draw_receipt("\u20b92,450.00")
save_jpeg(clean, f"{OUT}/clean_receipt.jpg",
          {271: "SamplePhone", 272: "SamplePhone X1", 305: "PaySample 1.0"},
          quality=95)
print("wrote", f"{OUT}/clean_receipt.jpg")


def donor_patch(text, w, h, fsize, anchor_xy, anchor):
    """Simulate a forger's source material: crisp digits from an image that has
    been through heavy (WhatsApp-style) JPEG compression."""
    p = Image.new("RGB", (w, h), (245, 247, 250))
    ImageDraw.Draw(p).text(anchor_xy, text, font=font(fsize), fill=INK, anchor=anchor)
    p.save("/tmp/_donor.jpg", "JPEG", quality=50)
    return Image.open("/tmp/_donor.jpg")


# 2) tampered — the forger's realistic workflow:
#    open the original -> paste amount digits lifted from a heavily-compressed
#    image (different quantization history) -> re-save. Real pixel compositing.
tam = Image.open(f"{OUT}/clean_receipt.jpg").convert("RGB")
tam.paste(donor_patch("\u20b99,850.00", 420, 100, 72, (210, 50), "mm"), (60, 430))
ref_patch = donor_patch("418229739999", 190, 40, 24, (120, 20), "rm")
tam.paste(ref_patch, (280, 652))
save_jpeg(tam, f"{OUT}/tampered_receipt.jpg",
          {271: "SamplePhone", 272: "SamplePhone X1", 305: "PaySample 1.0"},
          quality=95)
print("wrote", f"{OUT}/tampered_receipt.jpg",
      "(amount 2,450 -> 9,850 and ref digits composited from q50-compressed donors)")
