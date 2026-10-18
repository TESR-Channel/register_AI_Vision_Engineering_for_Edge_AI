"""Chapter 3, page 2 - "What You'll Be Able to Do", as a grouped checklist.

The page used to be two columns of ten equal-weight lines at 28px, which gave
the eye nothing to grab: no grouping, no hierarchy, and a third of the slide
left empty underneath. Same ten items here, sorted into four named groups, set
at 34px with a gold tick against each one, filling the figure box.

Nothing here is a result - it is the chapter's contents page, and every item
is a tab that exists in the TESR Image Processing Lab.

Output: /home/claude/deck/img/ch3_outcomes.png, for the 1676x620 figure box.
"""
import os
import sys

sys.path.insert(0, "/home/claude/deck")
from ip_panels2 import fig, t, OUT, GOLD, WHITE, MUTED

W, H = 1676, 620

GROUPS = [
    ("Get a picture on screen", [
        "Open a USB camera and see live video",
        "Draw boxes, circles, lines and text on the image",
        "Measure the FPS of your own loop",
    ]),
    # "count" lives in the heading, so the items need not repeat it - which
    # also keeps the longest line clear of the right-hand column
    ("Separate and count", [
        "Separate objects from the background",
        "Pick out one colour and count only that colour",
    ]),
    ("Find shapes", [
        "Find circles, such as coins or round parts",
        "Find rectangles and read off their tilt in degrees",
        "Find straight lines at the angles you choose",
    ]),
    ("Read what is in the frame", [
        "Find faces, eyes and smiles",
        "Read ArUco, QR and barcodes",
    ]),
]

HEAD_PX, ITEM_PX = 38, 34
LINE, GAP, HEAD_DROP = 58, 52, 64


def tick(a, x, y, r=13, lw=3.4):
    """A gold check mark, drawn rather than typed so no font has to carry it."""
    a.plot([x - r * 0.72, x - r * 0.12, x + r * 0.82],
           [y + r * 0.04, y + r * 0.62, y - r * 0.66],
           color=GOLD, lw=lw * 2, solid_capstyle="round",
           solid_joinstyle="round", zorder=8)


def column(a, x, groups, top):
    y = top
    for i, (head, items) in enumerate(groups):
        if i:
            y += GAP
        t(a, x, y, head.upper(), px=HEAD_PX, c=GOLD, w="bold")
        y += HEAD_DROP
        for item in items:
            tick(a, x + 13, y)
            t(a, x + 52, y, item, px=ITEM_PX, c=WHITE)
            y += LINE
        y -= LINE
        y += LINE
    return y


def block_height(groups):
    h = 0
    for i, (_, items) in enumerate(groups):
        if i:
            h += GAP
        h += HEAD_DROP + LINE * len(items)
    return h - LINE + ITEM_PX


def panel_outcomes():
    f, a = fig(W, H)
    left, right = GROUPS[:2], GROUPS[2:]
    tallest = max(block_height(left), block_height(right))
    top = (H - tallest) / 2 + ITEM_PX / 2
    column(a, 40, left, top)
    column(a, 876, right, top)
    f.savefig(OUT + "ch3_outcomes.png", dpi=100, facecolor="#0F1216")
    print("ch3_outcomes.png  ·  %d items in %d groups  ·  items %dpx, headings %dpx"
          % (sum(len(i) for _, i in GROUPS), len(GROUPS), ITEM_PX, HEAD_PX))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    panel_outcomes()
