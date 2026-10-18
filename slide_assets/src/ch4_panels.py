"""Figures for chapter 4, AI and Deep Learning fundamentals.

The chapter ran nine lecture pages before a learner touched anything, and
four of those pages said on their own face that the learner would never do
the thing being taught ("You will not type this", "the framework does it",
"You will not design one"). This replaces five of them with one figure.

The workshop also starts in the TESR AI Vision Web Trainer rather than at a
terminal, so the second figure is the browser-first path: collect and label
in the page, train and test there, then take the dataset to the PC.

The loss curves are drawn shapes, not a measured run - the figure says so on
its own face. Everything else here is a diagram of a workflow.

Output: /home/claude/deck/img/ch4_*.png, for a 1920x1080 slide at 2x.
"""
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "/home/claude/deck")
from ch0_panels import font, wrap, OUT, BG, PANEL, EDGE, GOLD, MUTED, WHITE, S, W, H

GREEN = (111, 191, 115)
BLUE = (91, 141, 239)
DIM = (108, 116, 128)


def canvas():
    img = Image.new("RGB", (W * S, H * S), BG)
    return img, ImageDraw.Draw(img)


def panel(d, x, y, w, h, ec=EDGE):
    d.rounded_rectangle([x * S, y * S, (x + w) * S, (y + h) * S],
                        radius=12 * S, fill=PANEL, outline=ec, width=2)


# --------------------------------------------------------------------------
# ch4_training - what training actually does, in one picture
# --------------------------------------------------------------------------
def fig_training():
    img, d = canvas()
    f_h = font("Kanit", 28, "bold")
    f_b = font("Kanit", 24)
    f_s = font("Kanit", 22)
    f_n = font("Kanit", 24)

    PAD, gap = 14, 26
    cw = (W - 2 * PAD - 2 * gap) // 3
    top, chh = 46, 452

    # ---- 1. the model is numbers ----
    x = PAD
    panel(d, x, top, cw, chh)
    d.text(((x + 26) * S, (top + 22) * S), "1   It is numbers, not rules",
           font=f_h, fill=GOLD)
    cols = [3, 5, 5, 2]
    cx0, cy0 = x + 70, top + 110
    colgap, noderad = 108, 11
    pts = []
    for ci, n in enumerate(cols):
        colx = cx0 + ci * colgap
        span = 44
        ys = [cy0 + 100 + (i - (n - 1) / 2) * span for i in range(n)]
        pts.append([(colx, yy) for yy in ys])
    for a, b in zip(pts, pts[1:]):
        for (x1, y1) in a:
            for (x2, y2) in b:
                d.line([x1 * S, y1 * S, x2 * S, y2 * S], fill=(46, 54, 64),
                       width=2)
    for ci, col in enumerate(pts):
        c = GOLD if ci in (0, len(pts) - 1) else BLUE
        for (px, py) in col:
            d.ellipse([(px - noderad) * S, (py - noderad) * S,
                       (px + noderad) * S, (py + noderad) * S],
                      fill=PANEL, outline=c, width=3)
    # what each column of the stack has worked out by that point - the part
    # of the old architecture slide that was actually worth keeping
    stage = ["pixels", "edges", "shapes", "answer"]
    for ci, name in enumerate(stage):
        lx = cx0 + ci * colgap - 26
        d.text((lx * S, (top + 312) * S), name, font=f_s,
               fill=GOLD if ci == len(stage) - 1 else MUTED)
        if ci < len(stage) - 1:
            ax = (lx + 76) * S
            ay = (top + 322) * S
            d.polygon([(ax, ay - 7 * S), (ax + 13 * S, ay),
                       (ax, ay + 7 * S)], fill=(70, 80, 92))
    for i, ln in enumerate(["each layer works from",
                            "what the one before found",
                            "- nobody writes those numbers"]):
        d.text(((x + 26) * S, (top + 356 + i * 34) * S), ln, font=f_b,
               fill=WHITE if i == 0 else MUTED)

    # ---- 2. the loop ----
    x += cw + gap
    panel(d, x, top, cw, chh)
    d.text(((x + 26) * S, (top + 22) * S), "2   One loop, over and over",
           font=f_h, fill=GOLD)
    steps = ["show a batch of images",
             "compare answer to label",
             "nudge every weight",
             "repeat"]
    for i, s in enumerate(steps):
        by = top + 86 + i * 62
        bh = 50
        d.rounded_rectangle([(x + 26) * S, by * S, (x + cw - 26) * S,
                             (by + bh) * S], radius=8 * S,
                            fill=(24, 29, 36), outline=EDGE, width=2)
        d.ellipse([(x + 42) * S, (by + 12) * S, (x + 68) * S, (by + 38) * S],
                  fill=GOLD)
        d.text(((x + 51) * S, (by + 14) * S), str(i + 1), font=f_s,
               fill=(14, 17, 22))
        d.text(((x + 82) * S, (by + 12) * S), s, font=f_b, fill=WHITE)
        if i < len(steps) - 1:
            ax = (x + cw // 2) * S
            d.polygon([(ax - 9 * S, (by + bh + 2) * S),
                       (ax + 9 * S, (by + bh + 2) * S),
                       (ax, (by + bh + 11) * S)], fill=GOLD)
    for i, ln in enumerate(["you never write this loop",
                            "the trainer runs it for you"]):
        d.text(((x + 26) * S, (top + 356 + i * 34) * S), ln, font=f_b,
               fill=WHITE if i == 0 else MUTED)

    # ---- 3. the loss ----
    x += cw + gap
    panel(d, x, top, cw, chh)
    d.text(((x + 26) * S, (top + 22) * S), "3   One number tells you how it goes",
           font=f_h, fill=GOLD)
    gx, gy, gw, gh = x + 42, top + 86, cw - 84, 174
    d.rectangle([gx * S, gy * S, (gx + gw) * S, (gy + gh) * S],
                outline=EDGE, width=2)
    n = 90
    xs = np.linspace(0, 1, n)
    # loss falls; the validation line turns back up once it starts memorising
    train = 0.92 * np.exp(-3.1 * xs) + 0.06
    val = 0.92 * np.exp(-3.6 * xs) + 0.12 + 0.62 * np.clip(xs - 0.46, 0, 1) ** 1.3
    top_val = max(train.max(), val.max())

    def poly(series, colour, width=4):
        pts = [((gx + 10 + v * (gw - 20)) * S,
                (gy + gh - 12 - (sv / top_val) * (gh - 26)) * S)
               for v, sv in zip(xs, series)]
        d.line(pts, fill=colour, width=width, joint="curve")

    poly(train, GREEN)
    poly(val, (214, 92, 88))
    d.text(((gx + gw - 86) * S, (gy + gh - 32) * S), "epochs", font=f_s,
           fill=DIM)
    d.text(((x + 26) * S, (top + 282) * S), "training loss",
           font=f_b, fill=GREEN)
    d.text(((x + 26) * S, (top + 316) * S), "validation loss",
           font=f_b, fill=(214, 92, 88))
    for i, ln in enumerate(["when the red line turns",
                            "back up it is memorising",
                            "- stop there"]):
        d.text(((x + 26) * S, (top + 358 + i * 32) * S), ln, font=f_s,
               fill=MUTED)

    note = ("The curves are drawn to show the shape to look for, not a "
            "measured run - a real one is noisier. Everything a learner "
            "actually types is in the workshop that follows.")
    ny = H - 76
    for ln in wrap(d, note, f_n, (W - 2 * PAD) * S):
        d.text((PAD * S, ny * S), ln, font=f_n, fill=MUTED)
        ny += 34
    img.save(OUT + "ch4_training.png")
    print("ch4_training.png")


# --------------------------------------------------------------------------
# ch4_trainer - the browser-first workshop path
# --------------------------------------------------------------------------
STEPS = [
    ("1", "Collect", ["hold Capture", "move the object"], GOLD),
    ("2", "Label", ["name the classes", "drag a box"], GOLD),
    ("3", "Train here", ["trains in seconds", "nothing installed"], BLUE),
    ("4", "Live test", ["point the webcam", "see it answer"], BLUE),
    ("5", "Download", ["the YOLO dataset", "photos + labels"], GREEN),
    ("6", "Train on the PC", ["python train.py", "a real YOLO model"], GREEN),
]


def fig_trainer():
    img, d = canvas()
    f_t = font("Kanit", 32, "bold")
    f_sub = font("Kanit", 23)
    f_h = font("Kanit", 25, "bold")
    f_b = font("Kanit", 22)
    f_n = font("Kanit", 24)

    d.text((14 * S, 16 * S), "The browser does the first four steps",
           font=f_t, fill=WHITE)
    d.text((14 * S, 62 * S),
           "nothing to install, and your photos never leave your machine",
           font=f_sub, fill=MUTED)

    PAD, gap, arrow = 14, 10, 22
    n = len(STEPS)
    bw = (W - 2 * PAD - (n - 1) * (gap + arrow)) // n
    bh, top = 250, 112
    x = PAD
    for num, title, body, col in STEPS:
        d.rounded_rectangle([x * S, top * S, (x + bw) * S, (top + bh) * S],
                            radius=12 * S, fill=PANEL, outline=col, width=2)
        d.ellipse([(x + 20) * S, (top + 20) * S, (x + 56) * S, (top + 56) * S],
                  fill=col)
        d.text(((x + 32) * S, (top + 25) * S), num, font=f_h, fill=(14, 17, 22))
        d.text(((x + 20) * S, (top + 74) * S), title, font=f_h, fill=WHITE)
        yy = top + 120
        for line in body:
            for seg in wrap(d, line, f_b, (bw - 40) * S):
                d.text(((x + 20) * S, yy * S), seg, font=f_b, fill=MUTED)
                yy += 30
        if num != str(n):
            ax = (x + bw + gap // 2 + 2) * S
            ay = (top + bh // 2) * S
            d.polygon([(ax, ay - 12 * S), (ax + 20 * S, ay),
                       (ax, ay + 12 * S)], fill=GOLD)
        x += bw + gap + arrow

    # what each half gives you
    y2 = top + bh + 28
    for i, (head, body, col) in enumerate([
            ("In the browser",
             "a working model in minutes, and a clear look at what your "
             "data is still missing", BLUE),
            ("On the PC",
             "the same photos and labels, trained properly - this is the "
             "model that ships", GREEN)]):
        bx_ = PAD + i * (W // 2 - 6)
        bwid = W // 2 - 20
        d.rounded_rectangle([bx_ * S, y2 * S, (bx_ + bwid) * S, (y2 + 134) * S],
                            radius=12 * S, fill=(24, 29, 36), outline=col,
                            width=2)
        d.text(((bx_ + 24) * S, (y2 + 18) * S), head, font=f_h, fill=col)
        yy = y2 + 56
        for seg in wrap(d, body, f_b, (bwid - 48) * S):
            d.text(((bx_ + 24) * S, yy * S), seg, font=f_b, fill=MUTED)
            yy += 30

    d.text((PAD * S, (H - 36) * S),
           "Steps and button names follow the Web Trainer's own README - "
           "check them against the live page before the class.",
           font=f_n, fill=MUTED)
    img.save(OUT + "ch4_trainer.png")
    print("ch4_trainer.png")


if __name__ == "__main__":
    fig_training()
    fig_trainer()
