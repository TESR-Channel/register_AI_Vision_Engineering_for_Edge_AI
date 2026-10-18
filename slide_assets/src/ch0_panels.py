"""Figures for chapter 0, the course overview.

ch0_tools - the three TESR browser tools the course is built around.

Every claim on the card comes from that tool's own README, not from memory:
what it does, what it can read, and the handoff button between two of them.
No benchmark, accuracy or availability claim appears anywhere.

Output: /home/claude/deck/img/ch0_*.png, for a 1920x1080 slide at 2x.
"""
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm

OUT = "/home/claude/deck/img/"

BG = (15, 18, 22)
PANEL = (27, 32, 39)
EDGE = (58, 67, 79)
GOLD = (217, 164, 65)
MUTED = (154, 163, 174)
WHITE = (240, 240, 242)

S = 2
W, H = 1676, 620


def font(name, px, weight="normal"):
    p = fm.findfont(fm.FontProperties(family=name, weight=weight),
                    fallback_to_default=True)
    return ImageFont.truetype(p, int(px * S))


def wrap(d, text, fnt, maxw):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=fnt) <= maxw or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


TOOLS = [
    ("TESR Image Processing Lab",
     "Tune OpenCV live in the page, then take the Python with you.",
     ["circles · colour · lines · shapes",
      "faces, eyes and smiles",
      "ArUco, QR and five barcode types"],
     "tesr-channel.github.io/image_processing_lab"),
    ("TESR AI Vision Web Trainer",
     "Capture, label and train a model in the browser. Nothing installed.",
     ["detection, including rotated boxes",
      "classification, no boxes needed",
      "live webcam test before you export"],
     "tesr-channel.github.io/AI_vision_Trainer"),
    ("TESR ROI Studio",
     "Mark the regions you care about on a camera view, get Python back.",
     ["7-segment, OCR, licence plate",
      "barcode and QR, colour, presence",
      "level, measurement, counting lines"],
     "tesr-channel.github.io/AI_ROI_Designer"),
]


def fig_tools():
    img = Image.new("RGB", (W * S, H * S), BG)
    d = ImageDraw.Draw(img)

    f_t = font("Kanit", 31, "bold")
    f_w = font("Kanit", 25)
    f_b = font("Kanit", 24)
    f_u = font("DejaVu Sans Mono", 17)
    f_n = font("Kanit", 26)

    PAD, gap = 14, 30
    cw = (W - 2 * PAD - 2 * gap) // 3
    top, ch = 18, 418

    x = PAD
    for name, what, bullets, url in TOOLS:
        d.rounded_rectangle([x * S, top * S, (x + cw) * S, (top + ch) * S],
                            radius=12 * S, fill=PANEL, outline=EDGE, width=2)
        M = 28
        y = top + 26
        for seg in wrap(d, name, f_t, (cw - 2 * M) * S):
            d.text(((x + M) * S, y * S), seg, font=f_t, fill=GOLD)
            y += 42
        y += 8
        for seg in wrap(d, what, f_w, (cw - 2 * M) * S):
            d.text(((x + M) * S, y * S), seg, font=f_w, fill=WHITE)
            y += 34
        y += 14
        for b in bullets:
            d.text(((x + M) * S, y * S), "•", font=f_b, fill=GOLD)
            for i, seg in enumerate(wrap(d, b, f_b, (cw - 2 * M - 24) * S)):
                d.text(((x + M + 24) * S, y * S), seg, font=f_b, fill=MUTED)
                y += 32
            y += 6
        d.text(((x + M) * S, (top + ch - 44) * S), url, font=f_u, fill=GOLD)
        x += cw + gap

    # the one real handoff between two of the tools
    y1 = top + ch + 30
    d.text((PAD * S, y1 * S),
           "The Trainer's  Send to ROI Studio  button drops your trained "
           "model straight into ROI Studio.", font=f_n, fill=GOLD)
    y2 = y1 + 46
    for seg in wrap(d,
                    "All three run entirely in the browser - no install, no "
                    "server, no image leaves the machine - and all three hand "
                    "you code or a model to run on a PC, a Raspberry Pi or an "
                    "NVIDIA Jetson.", f_b, (W - 2 * PAD) * S):
        d.text((PAD * S, y2 * S), seg, font=f_b, fill=MUTED)
        y2 += 34

    img.save(OUT + "ch0_tools.png")
    print("ch0_tools.png")


if __name__ == "__main__":
    fig_tools()
