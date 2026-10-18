"""Chapter 4, page 4 - what a pretrained model is, shown rather than diagrammed.

The page used to be two stacks of unlabelled blocks with an arrow between them.
It is accurate and it teaches nobody what a pretrained model actually is.

Two of the three panels here are real YOLO11n runs, measured at render time and
printed to stdout:

  panel 1  the course's street photo   -> what the pretrained model knows
  panel 2  the course's own parts photo -> what it does when the class is not
                                           in its vocabulary

Panel 2 is the teaching moment and it is not staged: the model has no word for
a coloured disc, so it reaches for the nearest thing it was taught. The third
panel is a diagram of which part is kept and which is replaced, and says so.

Type sizes follow the course rule that figure text must not be smaller than the
slide's own body text: headings 32px, result lines 28px, captions 24px.

Output: /home/claude/deck/img/ch4_pretrained.png, for a 1713.6x540 figure box.
"""
import os
import sys
import textwrap

import cv2
import numpy as np

sys.path.insert(0, "/home/claude/deck")
from ip_panels2 import (fig, t, bx, put_img, OUT, GOLD, WHITE, MUTED, GREEN,
                        CRIM, PANEL, EDGE, BG)

from ultralytics import YOLO

W, H = 1714, 540
SAMPLES = "/home/claude/image_processing_lab/samples/"
AI = "/home/claude/deck/ai_samples/"

HEAD_PX, RESULT_PX, CAP_PX = 32, 28, 24
COLW, GAP, MARGIN = 538, 30, 20


def detect(path, weights="yolo11n.pt", square=False):
    """Run the stock model and draw its own output. Nothing is invented."""
    img = cv2.imread(path)
    if img is None:
        raise SystemExit("missing sample: " + path)
    if square:
        # same crop chapter 5 uses, so the two photo panels sit at one size
        img = img[200:1010, :].copy()
    res = YOLO(weights)(img, verbose=False)[0]
    counts = {}
    out = img.copy()
    for b in res.boxes:
        name = res.names[int(b.cls)]
        counts[name] = counts.get(name, 0) + 1
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
        col = (96, 191, 111) if name in ("bus", "person") else (88, 92, 214)
        th = max(2, int(round(img.shape[1] / 420)))
        cv2.rectangle(out, (x1, y1), (x2, y2), col, th)
        lab = "%s %.2f" % (name, float(b.conf))
        fs = img.shape[1] / 1100.0
        (tw, thh), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        cv2.rectangle(out, (x1, max(0, y1 - thh - 10)), (x1 + tw + 10, y1), col, -1)
        cv2.putText(out, lab, (x1 + 5, max(thh + 2, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (12, 15, 19), 2, cv2.LINE_AA)
    return out, counts


def tally(counts):
    if not counts:
        return "nothing found at all"
    return " · ".join("%s%s" % (k, " ×%d" % v if v > 1 else "")
                      for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))


def caption(a, x, y, s, width=COLW, px=CAP_PX, c=MUTED):
    for i, line in enumerate(textwrap.wrap(s, width=int(width / (px * 0.47)))):
        t(a, x, y + i * (px + 8), line, px=px, c=c)


def stack(a, x, y, w):
    """The diagram panel: which part is kept, which is replaced."""
    head_h, body_h = 74, 196
    bx(a, x, y, w, head_h, fc="#2a1a1a", ec=CRIM, lw=2, r=10)
    t(a, x + w / 2, y + 26, "HEAD", px=26, c=CRIM, ha="center", w="bold")
    t(a, x + w / 2, y + 54, "replaced with your classes", px=22, c=MUTED,
      ha="center")
    bx(a, x, y + head_h + 12, w, body_h, fc="#1d2128", ec=GOLD, lw=2, r=10)
    t(a, x + w / 2, y + head_h + 54, "BACKBONE", px=26, c=GOLD, ha="center",
      w="bold")
    t(a, x + w / 2, y + head_h + 92, "kept exactly as it is", px=22, c=MUTED,
      ha="center")
    t(a, x + w / 2, y + head_h + 140, "edges · texture · shape", px=22,
      c=WHITE, ha="center")
    t(a, x + w / 2, y + head_h + 176, "already learned, not retrained", px=21,
      c=MUTED, ha="center")


def panel_pretrained():
    f, a = fig(W, H)
    x = [MARGIN, MARGIN + COLW + GAP, MARGIN + 2 * (COLW + GAP)]

    street, c1 = detect(AI + "bus.jpg", square=True)
    parts, c2 = detect(SAMPLES + "ex2.jpg")
    _, c3 = detect(SAMPLES + "ex3.png")

    # ---- 1. what it already knows
    t(a, x[0], 26, "What it already knows", px=HEAD_PX, c=GOLD, w="bold")
    put_img(a, street, x[0], 56, COLW, 296)
    t(a, x[0], 382, tally(c1), px=RESULT_PX, c=GREEN)
    caption(a, x[0], 420,
            "Eighty everyday classes, learned from millions of photos you "
            "never had to collect.")

    # ---- 2. what it does with your parts
    t(a, x[1], 26, "What it does with your parts", px=HEAD_PX, c=GOLD,
      w="bold")
    put_img(a, parts, x[1], 56, COLW, 296)
    t(a, x[1], 382, tally(c2), px=RESULT_PX, c=CRIM)
    caption(a, x[1], 420,
            "Same model, a tray of plain parts. It has no word for them, so "
            "it reaches for the nearest thing it knows.")

    # ---- 3. what you actually change
    t(a, x[2], 26, "What you actually change", px=HEAD_PX, c=GOLD, w="bold")
    stack(a, x[2], 56, COLW)
    t(a, x[2], 382, "diagram — not a measurement", px=22, c=MUTED)
    caption(a, x[2], 420,
            "You supply the class names and a few hundred photos. The part "
            "that already sees stays.")

    f.savefig(OUT + "ch4_pretrained.png", dpi=100, facecolor=BG)
    print("ch4_pretrained.png")
    print("  street photo  ->", c1 or "nothing")
    print("  parts ex2.jpg ->", c2 or "nothing")
    print("  parts ex3.png ->", c3 or "nothing  (the empty-list case)")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    panel_pretrained()
