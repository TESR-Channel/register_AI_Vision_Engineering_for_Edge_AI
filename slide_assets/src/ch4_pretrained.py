"""Chapter 4, page 4 - transfer learning, drawn the way the reference explains it.

Two earlier attempts failed for the same underlying reason: they tried to teach
the idea from detection results instead of from the structure of the model.
The first used a different photo per panel and read as unrelated claims; the
second used one photo of coins, and Anon's verdict was that it is hard to see
what is even going on.

So this version drops the photographs as the carrier and draws the network,
following Roboflow's "What Is Transfer Learning?" (roboflow.com/blog/
what-is-transfer-learning) and their video of the same name, which Anon gave as
the reference. Their sequence is: take a pretrained model, remove its
classifier and output layer, add new ones for your classes, freeze the feature
extractor, and train only the new layers.

Two rows, the same four blocks in each, so the eye can compare them straight
down the page:

  the model you download   image -> feature extractor -> classifier -> output
  the model you train      image -> feature extractor -> NEW        -> NEW

What changes between the rows is the whole lesson. The feature extractor is
identical and frozen; the last two blocks are thrown away and rebuilt for your
classes. The classes shown are bus and person, the same worked example the
chapter 6 workshop uses.

This is a diagram of how the model is put together, not a measurement, and it
says so on its face. The thumbnails are real course sample images.

Output: /home/claude/deck/img/ch4_pretrained.png, for a 1713.6x540 figure box.
"""
import os
import sys

import cv2

sys.path.insert(0, "/home/claude/deck")
from ip_panels2 import fig, t, bx, put_img, OUT, GOLD, WHITE, MUTED, GREEN, \
    CRIM, EDGE, BG

W, H = 1714, 540
AI = "/home/claude/deck/ai_samples/"
SAMPLES = "/home/claude/image_processing_lab/samples/"

ROW_PX, BLOCK_PX, SUB_PX, TAG_PX, NOTE_PX = 32, 28, 22, 22, 26

# x, width for the four stages, shared by both rows
THUMB = (60, 150)
FEAT = (266, 470)
HEAD = (790, 270)
OUTP = (1114, 380)
BH = 112                      # block height


def arrow(a, x0, x1, y, c=MUTED):
    a.annotate("", xy=(x1, y), xytext=(x0, y),
               arrowprops=dict(arrowstyle="-|>", color=c, lw=2.6,
                               mutation_scale=22, shrinkA=0, shrinkB=0),
               zorder=9)


def block(a, x, w, y, title, sub, edge, title_c=WHITE, fill="#1b2027"):
    bx(a, x, y, w, BH, fc=fill, ec=edge, lw=2, r=10)
    t(a, x + w / 2, y + 40, title, px=BLOCK_PX, c=title_c, ha="center",
      w="bold")
    t(a, x + w / 2, y + 78, sub, px=SUB_PX, c=MUTED, ha="center")


def thumb(a, path, x, w, y, crop=None):
    img = cv2.imread(path)
    if img is None:
        raise SystemExit("missing sample: " + path)
    if crop:
        img = img[crop[0]:crop[1], crop[2]:crop[3]].copy()
    put_img(a, img, x, y, w, BH)


def row(a, y, label, label_c, feat_sub, head_title, head_sub, out_title,
        out_sub, new_edge, new_title_c, thumb_path, thumb_caption, crop=None,
        tag=None, tag_c=CRIM):
    t(a, THUMB[0], y - 36, label, px=ROW_PX, c=label_c, w="bold")

    thumb(a, thumb_path, THUMB[0], THUMB[1], y, crop)
    t(a, THUMB[0] + THUMB[1] / 2, y + BH + 22, thumb_caption, px=SUB_PX,
      c=MUTED, ha="center")

    arrow(a, THUMB[0] + THUMB[1] + 16, FEAT[0] - 16, y + BH / 2)
    block(a, FEAT[0], FEAT[1], y, "Feature extractor", feat_sub, GOLD,
          title_c=GOLD)

    arrow(a, FEAT[0] + FEAT[1] + 16, HEAD[0] - 16, y + BH / 2)
    block(a, HEAD[0], HEAD[1], y, head_title, head_sub, new_edge,
          title_c=new_title_c)

    arrow(a, HEAD[0] + HEAD[1] + 16, OUTP[0] - 16, y + BH / 2)
    block(a, OUTP[0], OUTP[1], y, out_title, out_sub, new_edge,
          title_c=new_title_c)

    if tag:
        t(a, OUTP[0] + OUTP[1], y - 14, tag, px=TAG_PX, c=tag_c, ha="right",
          w="bold")


def panel_pretrained():
    f, a = fig(W, H)

    row(a, 78,
        "The model you download", MUTED,
        "finds edges, corners and shapes",
        "Classifier", "trained on 80 everyday names",
        "Output layer", "80 outputs",
        EDGE, MUTED,
        AI + "coffee.jpg", "millions of images",
        tag="these two are thrown away", tag_c=CRIM)

    row(a, 320,
        "The model you train", WHITE,
        "frozen — kept exactly as it is",
        "New classifier", "learns from your photos",
        "New output layer", "one per class: bus, person",
        GREEN, GREEN,
        AI + "bus.jpg", "your few hundred photos",
        crop=(200, 1010, 0, 810),
        tag="only these two are trained", tag_c=GREEN)

    t(a, THUMB[0], 500,
      "Freezing the feature extractor is the step that matters — without it, "
      "the model forgets what it already knew.", px=NOTE_PX, c=WHITE)
    t(a, W - 60, 500, "diagram — not a measurement", px=SUB_PX, c=MUTED,
      ha="right")

    f.savefig(OUT + "ch4_pretrained.png", dpi=100, facecolor=BG)
    print("ch4_pretrained.png")
    print("  structure follows roboflow.com/blog/what-is-transfer-learning")
    print("  stages: pretrained -> remove classifier + output -> add new ->"
          " freeze extractor -> train new layers")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    panel_pretrained()
