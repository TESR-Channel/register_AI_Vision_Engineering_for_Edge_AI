"""Chapter 4, page 4 - what a pretrained model is, shown on one scene.

The page used to be two stacks of unlabelled blocks with an arrow between them.
A first rebuild replaced it with real detections, but used a different photo in
each panel, so it read as two unrelated things rather than one model meeting
two situations. Anon's words: it looks like it has nothing to do with itself.

So: one photograph, carried through all three panels.

The photograph is the course's own ex3.jpg - eight coins on a plain surface -
and YOLO11n's behaviour on it is the whole lesson, measured at render time:

  it draws a box around all eight coins, correctly placed
  it then calls seven of them "cake" and one "orange"

The finding already works. Only the naming is wrong. That is exactly why you
keep the backbone and replace the head, and a learner can see it in one glance
instead of being told it.

Panel 1 shows the boxes with no labels, panel 2 reveals the labels on the same
boxes, panel 3 is the keep-and-replace structure, marked on its face as a
diagram rather than a measurement.

Output: /home/claude/deck/img/ch4_pretrained.png, for a 1713.6x540 figure box.
"""
import os
import sys
import textwrap

import cv2

sys.path.insert(0, "/home/claude/deck")
from ip_panels2 import (fig, t, bx, put_img, OUT, GOLD, WHITE, MUTED, GREEN,
                        CRIM, EDGE, BG)

from ultralytics import YOLO

W, H = 1714, 540
SCENE = "/home/claude/image_processing_lab/samples/ex3.jpg"
TRUTH = "coins"

HEAD_PX, RESULT_PX, CAP_PX = 32, 28, 24
COLW, GAP, MARGIN = 538, 30, 20

BOX_GREEN = (96, 191, 111)      # BGR - localisation, which is right
BOX_RED = (88, 92, 214)         # BGR - the names, which are not


def run(path, weights="yolo11n.pt"):
    img = cv2.imread(path)
    if img is None:
        raise SystemExit("missing sample: " + path)
    res = YOLO(weights)(img, verbose=False)[0]
    boxes, counts = [], {}
    for b in res.boxes:
        name = res.names[int(b.cls)]
        counts[name] = counts.get(name, 0) + 1
        boxes.append((name, float(b.conf), [int(v) for v in b.xyxy[0]]))
    return img, boxes, counts


def draw(img, boxes, colour, labels):
    """Redraw the model's own output. Boxes always; names only when asked."""
    out = img.copy()
    th = max(2, int(round(img.shape[1] / 330)))
    for name, conf, (x1, y1, x2, y2) in boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, th)
        if not labels:
            continue
        lab = "%s %.2f" % (name, conf)
        fs = img.shape[1] / 900.0
        (tw, thh), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        cv2.rectangle(out, (x1, max(0, y1 - thh - 12)),
                      (x1 + tw + 12, y1), colour, -1)
        cv2.putText(out, lab, (x1 + 6, max(thh + 2, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (12, 15, 19), 2, cv2.LINE_AA)
    return out


def tally(counts):
    return " · ".join("%s%s" % (k, " ×%d" % v if v > 1 else "")
                      for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))


def caption(a, x, y, s, width=COLW, px=CAP_PX, c=MUTED):
    for i, line in enumerate(textwrap.wrap(s, width=int(width / (px * 0.47)))):
        t(a, x, y + i * (px + 8), line, px=px, c=c)


def stack(a, x, y, w):
    """The diagram panel: which part is kept, which is replaced."""
    head_h, body_h = 88, 182
    bx(a, x, y, w, head_h, fc="#2a1a1a", ec=CRIM, lw=2, r=10)
    t(a, x + w / 2, y + 28, "HEAD", px=26, c=CRIM, ha="center", w="bold")
    t(a, x + w / 2, y + 62, "names what was found", px=22, c=MUTED,
      ha="center")
    bx(a, x, y + head_h + 14, w, body_h, fc="#1d2128", ec=GOLD, lw=2, r=10)
    t(a, x + w / 2, y + head_h + 54, "BACKBONE", px=26, c=GOLD, ha="center",
      w="bold")
    t(a, x + w / 2, y + head_h + 90, "finds the objects", px=22, c=MUTED,
      ha="center")
    t(a, x + w / 2, y + head_h + 134, "edges · texture · shape", px=22,
      c=WHITE, ha="center")
    t(a, x + w - 20, y + 24, "replace", px=21, c=CRIM, ha="right", w="bold")
    t(a, x + w - 20, y + head_h + 38, "keep", px=21, c=GOLD, ha="right",
      w="bold")


def panel_pretrained():
    f, a = fig(W, H)
    x = [MARGIN, MARGIN + COLW + GAP, MARGIN + 2 * (COLW + GAP)]

    img, boxes, counts = run(SCENE)
    n = len(boxes)

    # ---- 1. the finding, with the names withheld
    t(a, x[0], 26, "It finds every one of them", px=HEAD_PX, c=GOLD, w="bold")
    put_img(a, draw(img, boxes, BOX_GREEN, labels=False), x[0], 56, COLW, 296)
    t(a, x[0], 382, "%d objects, %d boxes" % (n, n), px=RESULT_PX, c=GREEN)
    caption(a, x[0], 420,
            "A stock model that has never been shown a coin. The finding "
            "already works.")

    # ---- 2. the same boxes, now with what it calls them
    t(a, x[1], 26, "It just has the wrong words", px=HEAD_PX, c=GOLD,
      w="bold")
    put_img(a, draw(img, boxes, BOX_RED, labels=True), x[1], 56, COLW, 296)
    t(a, x[1], 382, tally(counts), px=RESULT_PX, c=CRIM)
    caption(a, x[1], 420,
            "Its vocabulary is a fixed list of eighty everyday things. "
            "“%s” is not on it." % TRUTH[:-1].capitalize())

    # ---- 3. so you change only the naming part
    t(a, x[2], 26, "So you change only the words", px=HEAD_PX, c=GOLD,
      w="bold")
    stack(a, x[2], 56, COLW)
    t(a, x[2], 382, "diagram — not a measurement", px=22, c=MUTED)
    caption(a, x[2], 420,
            "A few hundred of your own photos retrain the naming. The seeing "
            "is already done.")

    f.savefig(OUT + "ch4_pretrained.png", dpi=100, facecolor=BG)
    print("ch4_pretrained.png")
    print("  scene      :", os.path.basename(SCENE), img.shape[1::-1])
    print("  boxes drawn:", n)
    print("  it calls them:", counts)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    panel_pretrained()
