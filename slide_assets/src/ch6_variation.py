"""Chapter 6, page 4 - what "vary your images" actually looks like.

The page showed five columns of blank grey rectangles labelled LIGHTING,
ANGLE, DISTANCE, POSITION, BACKGROUND. Anon's verdict was that it shows
nothing at all: blocks and words, no picture of the thing being described.

So this replaces it with five real frames of one subject, each altered the way
the column headings describe, so a learner can see what a varied set looks
like before being asked to shoot one.

Honesty: these five are **one photograph altered with OpenCV**, not five
separate photographs taken on a line. That is stated on the figure itself, in
the same way the rest of the course marks a drawn figure as a diagram. The
point of the slide is the idea of variation, and on their own line the learner
takes five real ones.

Subject is the course's own ex3.jpg - coins on a plain surface - because small,
round, near-identical parts photographed from above is what an inspection job
usually looks like.

Output: /home/claude/deck/img/ch6_variation.png, for a 1713.6x405.853 box.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/home/claude/deck")
from ip_panels2 import fig, t, put_img, OUT, GOLD, WHITE, MUTED, BG

W, H = 1714, 406
SCENE = "/home/claude/image_processing_lab/samples/ex3.jpg"

LABEL_PX, NOTE_PX = 26, 22
MARGIN, GAP, TH = 24, 16, 270


def surface(img):
    """The photo's own background colour, so filled areas are not flat white."""
    h, w = img.shape[:2]
    patch = img[int(h * 0.04):int(h * 0.16), int(w * 0.04):int(w * 0.16)]
    return [int(v) for v in patch.reshape(-1, 3).mean(axis=0)]


def dim(img, f=0.34):
    """Lights off: the whole frame loses exposure, it does not just darken."""
    return cv2.convertScaleAbs(img, alpha=f, beta=-8)


def tilt(img, deg=14):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=surface(img))


def further(img, f=0.55):
    """The part sits further from the camera, so it covers fewer pixels.

    The frame has to be filled out somehow - there is no more scene in the
    photograph than the photograph holds - so it is filled with the surface
    colour sampled from the picture itself rather than with flat white.
    """
    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w * f), int(h * f)),
                       interpolation=cv2.INTER_AREA)
    out = np.full_like(img, 0)
    out[:] = surface(img)
    y, x = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
    out[y:y + small.shape[0], x:x + small.shape[1]] = small
    return out


def at_edge(img, keep=0.68):
    """Crop a window off to one side, so the parts sit near the frame edge.

    A crop rather than a shift, so the frame is filled with real background
    instead of padding.
    """
    h, w = img.shape[:2]
    x0 = int(w * (1 - keep))
    return cv2.resize(img[:, x0:], (w, h), interpolation=cv2.INTER_AREA)


FRAMES = [
    ("as installed", lambda im: im),
    ("lights off", dim),
    ("camera tilted", tilt),
    ("part further away", further),
    ("part at the edge", at_edge),
]


def panel_variation():
    base = cv2.imread(SCENE)
    if base is None:
        raise SystemExit("missing sample: " + SCENE)

    f, a = fig(W, H)
    cw = (W - 2 * MARGIN - (len(FRAMES) - 1) * GAP) / len(FRAMES)

    for i, (label, fn) in enumerate(FRAMES):
        x = MARGIN + i * (cw + GAP)
        put_img(a, fn(base.copy()), x, 16, cw, TH)
        t(a, x + cw / 2, 308, label, px=LABEL_PX,
          c=GOLD if i else WHITE, ha="center", w="bold" if i == 0 else "normal")

    t(a, MARGIN, 364,
      "One photograph altered five ways, to show what a varied set looks like"
      " — on your own line you take five real ones.", px=NOTE_PX, c=MUTED)

    f.savefig(OUT + "ch6_variation.png", dpi=100, facecolor=BG)
    print("ch6_variation.png")
    print("  scene :", os.path.basename(SCENE), base.shape[1::-1])
    print("  frames:", ", ".join(n for n, _ in FRAMES))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    panel_variation()
