"""Chapter cover glyphs — one picture per chapter, all in the same frame.

Every chapter cover gets the same viewfinder plate used on the deck 0 cover,
with an illustration inside saying what that chapter is about. Rendered at 2x
the Canva rectangle (660 x 470 slide px).

Nothing here states a measurement: no FPS, accuracy, latency or price. These
are pictures of the job, not claims about it.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, Rectangle, Circle, Polygon,
                                FancyArrowPatch, Wedge, Ellipse)
import numpy as np
from PIL import Image

BG, PANEL, PANEL2, EDGE = "#0F1216", "#1B2027", "#161A20", "#3A434F"
GOLD, GOLD_D, CRIM = "#D9A441", "#A87C2C", "#D62828"
WHITE, MUTED, DIM = "#FFFFFF", "#9AA3AE", "#6C7480"
GREEN, BLUE, STEEL = "#6FBF73", "#5B8DEF", "#AEB5BD"
TH, MONO = "Kanit", "DejaVu Sans Mono"
S, P = 2, 0.72

W, H = 660, 470
PX, PY, PW, PH = 8, 6, 644, 386          # the plate
CAP_Y = 440                              # caption baseline, under the plate
OUT = "/home/claude/deck/img/"


# --------------------------------------------------------------- primitives
def t(a, x, y, s, px=18, c=WHITE, ha="left", va="center", w="normal", z=8,
      fam=TH, alpha=1, rot=0):
    a.text(x, y, s, fontsize=px * P * S, color=c, ha=ha, va=va, rotation=rot,
           fontfamily=fam, fontweight=w, zorder=z, alpha=alpha)


def bx(a, x, y, w, h, fc=PANEL, ec="none", lw=0, r=8, z=5, alpha=1,
       ls="solid"):
    a.add_patch(FancyBboxPatch((x, y), w, h,
                boxstyle="round,pad=0,rounding_size=%g" % r, fc=fc, ec=ec,
                lw=lw * S, zorder=z, alpha=alpha, linestyle=ls))


def rect(a, x, y, w, h, fc="none", ec=GOLD, lw=1.6, z=7, alpha=1, ls="solid"):
    a.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw * S, zorder=z,
                          alpha=alpha, linestyle=ls))


def arr(a, x0, y0, x1, y1, c=GOLD, lw=1.6, z=7, ls="solid", head=13):
    a.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                mutation_scale=head * S, color=c, lw=lw * S, zorder=z,
                linestyle=ls, shrinkA=0, shrinkB=0))


def ln(a, x0, y0, x1, y1, c=EDGE, lw=1.2, z=6, alpha=1, ls="solid"):
    a.plot([x0, x1], [y0, y1], color=c, lw=lw * S, zorder=z, alpha=alpha,
           ls=ls, solid_capstyle="round")


def dot(a, x, y, r, c, z=8, ec="none", lw=0):
    a.add_patch(Circle((x, y), r, fc=c, ec=ec, lw=lw * S, zorder=z))


def tag(a, x, y, label, c=GREEN, z=9, px=11):
    """A detection label tab sitting on top of a box."""
    tw = len(label) * 7.2 + 11
    a.add_patch(Rectangle((x, y - 17), tw, 16, fc=c, ec="none", zorder=z))
    t(a, x + 5.5, y - 8.6, label, px=px, c="#0E1116", w="bold", z=z + 1,
      fam=MONO)


def chip(a, x, y, label, w=None, c=GOLD, fc="#1A1410", px=12, z=7):
    w = w if w else len(label) * 8.2 + 24
    bx(a, x, y, w, 28, fc=fc, ec=GOLD_D, lw=1.2, r=7, z=z)
    t(a, x + w / 2, y + 14.5, label, px=px, c=c, ha="center", w="bold",
      z=z + 1, fam=MONO)
    return w


def img(a, arr2d, x0, y0, x1, y1, z=8):
    a.imshow(arr2d, extent=(x0, x1, y1, y0), cmap="gray", vmin=0, vmax=255,
             interpolation="nearest", zorder=z)


# ------------------------------------------------------------------- plate
def plate(caption):
    f = plt.figure(figsize=(W * S / 100, H * S / 100), dpi=100)
    f.patch.set_facecolor(BG)
    a = f.add_axes([0, 0, 1, 1])
    a.set_xlim(0, W); a.set_ylim(0, H); a.invert_yaxis()
    a.axis("off"); a.set_facecolor(BG)

    bx(a, PX, PY, PW, PH, fc="#10151B", ec=EDGE, lw=1.4, r=12, z=2)
    for gx in range(1, 14):
        ln(a, PX + gx * PW / 14, PY + 8, PX + gx * PW / 14, PY + PH - 8,
           c=EDGE, lw=0.7, z=3, alpha=0.20)
    for gy in range(1, 9):
        ln(a, PX + 8, PY + gy * PH / 9, PX + PW - 8, PY + gy * PH / 9,
           c=EDGE, lw=0.7, z=3, alpha=0.20)
    BL = 30
    for cx, cy, sx, sy in ((PX + 11, PY + 11, 1, 1),
                           (PX + PW - 11, PY + 11, -1, 1),
                           (PX + 11, PY + PH - 11, 1, -1),
                           (PX + PW - 11, PY + PH - 11, -1, -1)):
        ln(a, cx, cy, cx + sx * BL, cy, c=GOLD, lw=2.4, z=12)
        ln(a, cx, cy, cx, cy + sy * BL, c=GOLD, lw=2.4, z=12)

    t(a, PX + 2, CAP_Y, caption, px=19, c=MUTED, z=9)
    return f, a


# =========================================================== 01 environment
def g01(a):
    x, y, w, h = 92, 58, 476, 266
    bx(a, x, y, w, h, fc="#0A0D11", ec=EDGE, lw=1.4, r=10, z=5)
    bx(a, x, y, w, 36, fc="#1A1F26", ec="none", r=10, z=6)
    rect(a, x, y + 26, w, 10, fc="#1A1F26", ec="none", z=6)
    for i, c in enumerate(("#E05A4F", "#E0B441", "#5FB85F")):
        dot(a, x + 22 + i * 20, y + 18, 6, c, z=7)
    t(a, x + 96, y + 18.5, "Terminal", px=13, c=DIM, z=7, fam=MONO)

    rows = [("$ ", "python -m venv .venv", WHITE),
            ("$ ", "pip install -r requirements.txt", WHITE),
            ("$ ", "python verify_setup.py", WHITE),
            ("", "✓ python   ✓ opencv   ✓ camera", GREEN)]
    yy = y + 70
    for pre, s, c in rows:
        if pre:
            t(a, x + 24, yy, pre, px=15, c=GOLD, z=8, fam=MONO, w="bold")
        t(a, x + 24 + (16 if pre else 0), yy, s, px=15, c=c, z=8, fam=MONO)
        yy += 42
    ln(a, x + 24, y + 238, x + 34, y + 238, c=GOLD, lw=2.2, z=8)


# ============================================================== 02 workflow
def g02(a):
    labels = ["COLLECT", "LABEL", "TRAIN", "DEPLOY"]
    bw, gap, by, bh = 118, 32, 112, 72
    x0 = (W - (4 * bw + 3 * gap)) / 2
    for i, lab in enumerate(labels):
        x = x0 + i * (bw + gap)
        bx(a, x, by, bw, bh, fc=PANEL, ec=GOLD_D, lw=1.4, r=9, z=6)
        t(a, x + bw / 2, by + bh / 2, lab, px=14, c=GOLD, ha="center",
          w="bold", z=8, fam=MONO)
        if i < 3:
            arr(a, x + bw + 5, by + bh / 2, x + bw + gap - 5, by + bh / 2)
    xl, xr = x0 + bw / 2, x0 + 3 * (bw + gap) + bw / 2
    ln(a, xr, by + bh, xr, by + bh + 58, c=GOLD_D, lw=1.5, z=6)
    ln(a, xl, by + bh + 58, xr, by + bh + 58, c=GOLD_D, lw=1.5, z=6)
    arr(a, xl, by + bh + 58, xl, by + bh + 6, c=GOLD_D, lw=1.5)
    t(a, (xl + xr) / 2, by + bh + 80, "measure, then improve", px=14, c=MUTED,
      ha="center", z=8)


# ========================================================= 03 camera, image
def g03(a):
    cx, cy = 136, 186
    bx(a, cx - 60, cy - 44, 100, 88, fc=PANEL, ec=EDGE, lw=1.5, r=10, z=6)
    bx(a, cx - 50, cy - 59, 42, 17, fc=PANEL, ec=EDGE, lw=1.3, r=4, z=5)
    dot(a, cx + 40, cy, 38, PANEL2, z=7, ec=EDGE, lw=1.5)
    dot(a, cx + 40, cy, 29, "#0B0E12", z=8, ec=GOLD_D, lw=1.5)
    for k in range(6):
        a.add_patch(Wedge((cx + 40, cy), 27, 60 * k + 6, 60 * k + 54,
                          width=10, fc=GOLD, ec="none", alpha=0.55, zorder=9))
    dot(a, cx + 40, cy, 10, "#05070A", z=10, ec=GOLD, lw=1.3)
    arr(a, cx + 86, cy, 268, cy, lw=1.7)

    gx0, gy0, cell = 296, 92, 36
    rng = np.random.RandomState(5)
    v = rng.randint(55, 215, (6, 8))
    for r in range(6):
        for c in range(8):
            g = int(v[r, c])
            rect(a, gx0 + c * cell, gy0 + r * cell, cell, cell,
                 fc="#%02x%02x%02x" % (g, g, g), ec="#0F1216", lw=0.6, z=7)
    rect(a, gx0 + 2 * cell, gy0 + 2 * cell, 3 * cell, 3 * cell, ec=GOLD,
         lw=2.4, z=9)
    t(a, gx0 + 3.5 * cell, gy0 + 6 * cell + 22, "3 × 3 kernel", px=14,
      c=GOLD, ha="center", z=9, fam=MONO)
    t(a, gx0 + 3.5 * cell, gy0 - 20, "every pixel is a number", px=14,
      c=MUTED, ha="center", z=9)


# ============================================================ 04 neural net
def g04(a):
    cols = [(150, 4), (258, 6), (366, 6), (474, 3)]
    pts = []
    for x, n in cols:
        span = (n - 1) * 37
        pts.append([(x, 196 - span / 2 + i * 37) for i in range(n)])
    rng = np.random.RandomState(2)
    for i in range(len(pts) - 1):
        for (x0, y0) in pts[i]:
            for (x1, y1) in pts[i + 1]:
                ln(a, x0, y0, x1, y1, c=EDGE, lw=0.9, z=5,
                   alpha=0.25 + 0.45 * rng.rand())
    for i, col in enumerate(pts):
        c = GOLD if i in (0, len(pts) - 1) else STEEL
        for (x, y) in col:
            dot(a, x, y, 12, "#10151B", z=7, ec=c, lw=1.8)
            dot(a, x, y, 5.5, c, z=8)
    for x, lab in ((150, "input"), (312, "hidden layers"), (474, "output")):
        t(a, x, 324, lab, px=14, c=MUTED, ha="center", z=9)
    t(a, 330, 58, "weights are learned, not written", px=15, c=GOLD,
      ha="center", z=9)


# =============================================================== 05 detector
def g05(a):
    sx, sy, sw, sh = 92, 64, 476, 236
    bx(a, sx, sy, sw, sh, fc="#0C1014", ec=EDGE, lw=1.3, r=10, z=5)
    # a person
    px_, py_ = sx + 86, sy + 150
    dot(a, px_, py_ - 62, 15, STEEL, z=6)
    a.add_patch(Polygon([(px_ - 19, py_ + 56), (px_ - 13, py_ - 42),
                         (px_ + 13, py_ - 42), (px_ + 19, py_ + 56)],
                        closed=True, fc=STEEL, ec="none", zorder=6))
    rect(a, px_ - 34, py_ - 88, 68, 150, ec=GREEN, lw=1.9, z=8)
    tag(a, px_ - 34, py_ - 88, "person")
    # a box on the floor
    bx(a, sx + 190, sy + 128, 104, 72, fc="#C3C9D1", ec="#8C949E", lw=1.4,
       r=5, z=6)
    ln(a, sx + 190, sy + 152, sx + 294, sy + 152, c="#8C949E", lw=1.3, z=7)
    rect(a, sx + 182, sy + 120, 120, 88, ec=GREEN, lw=1.9, z=8)
    tag(a, sx + 182, sy + 120, "carton")
    # a bottle
    bbx, bby = sx + 364, sy + 118
    bx(a, bbx, bby + 20, 34, 66, fc="#9FB8A8", ec="#7D9487", lw=1.3, r=5, z=6)
    bx(a, bbx + 11, bby, 12, 24, fc="#9FB8A8", ec="#7D9487", lw=1.2, r=3, z=6)
    rect(a, bbx - 10, bby - 10, 54, 108, ec=GREEN, lw=1.9, z=8)
    tag(a, bbx - 10, bby - 10, "bottle")


# ======================================================= 06 your own images
def g06(a):
    bx(a, 120, 84, 232, 184, fc="#161B21", ec=EDGE, lw=1.2, r=8, z=4)
    bx(a, 110, 74, 232, 184, fc="#1A1F26", ec=EDGE, lw=1.2, r=8, z=5)
    tx, ty, tw, th = 100, 64, 232, 184
    bx(a, tx, ty, tw, th, fc="#20262E", ec=EDGE, lw=1.4, r=8, z=6)
    bx(a, tx + 56, ty + 74, 120, 72, fc="#C3C9D1", ec="#8C949E", lw=1.3, r=6,
       z=7)
    dot(a, tx + 86, ty + 110, 9, "#10151B", z=8, ec="#8C949E", lw=1.1)
    dot(a, tx + 146, ty + 110, 9, "#10151B", z=8, ec="#8C949E", lw=1.1)
    # the box being drawn
    rect(a, tx + 44, ty + 60, 144, 100, ec=GOLD, lw=2.0, z=9, ls=(0, (6, 4)))
    for hx, hy in ((tx + 44, ty + 60), (tx + 188, ty + 60),
                   (tx + 44, ty + 160), (tx + 188, ty + 160)):
        rect(a, hx - 5, hy - 5, 10, 10, fc=GOLD, ec="#0F1216", lw=0.8, z=10)
    tag(a, tx + 44, ty + 60, "part  ok", c=GOLD)

    arr(a, 360, 156, 404, 156, lw=1.7)
    bx(a, 416, 76, 164, 164, fc=PANEL, ec=GOLD_D, lw=1.4, r=10, z=6)
    t(a, 498, 104, "labels.txt", px=14, c=GOLD, ha="center", z=8, fam=MONO)
    ln(a, 436, 120, 560, 120, c=EDGE, lw=1.1, z=7)
    for i, s in enumerate(("0  0.42 0.51 0.30 0.27",
                           "0  0.71 0.44 0.18 0.22",
                           "1  0.22 0.63 0.25 0.19")):
        t(a, 432, 142 + i * 28, s, px=11, c=MUTED, z=8, fam=MONO)
    t(a, 498, 228, "one line per object", px=12, c=DIM, ha="center", z=8)
    t(a, 330, 300, "your images, your classes", px=15, c=MUTED, ha="center",
      z=9)


# =========================================================== 07 be accurate
def g07(a):
    # confusion matrix
    mx, my, cell = 106, 104, 72
    names = [["TP", "FN"], ["FP", "TN"]]
    tone = [["#223026", "#2C2420"], ["#2C2420", "#223026"]]
    edge = [[GREEN, CRIM], [CRIM, GREEN]]
    for r in range(2):
        for c in range(2):
            rect(a, mx + c * cell, my + r * cell, cell, cell,
                 fc=tone[r][c], ec=edge[r][c], lw=1.5, z=6)
            t(a, mx + c * cell + cell / 2, my + r * cell + cell / 2,
              names[r][c], px=17, c=edge[r][c], ha="center", w="bold", z=8,
              fam=MONO)
    t(a, mx + cell, my - 20, "predicted", px=13, c=MUTED, ha="center", z=8)
    t(a, mx - 18, my + cell, "actual", px=13, c=MUTED, ha="center", z=8,
      rot=90)

    # the curve
    gx, gy, gw, gh = 320, 92, 236, 160
    ln(a, gx, gy + gh, gx + gw, gy + gh, c=EDGE, lw=1.3, z=6)
    ln(a, gx, gy, gx, gy + gh, c=EDGE, lw=1.3, z=6)
    xs = np.linspace(0, 1, 40)
    ys = 1 - np.exp(-3.1 * xs)
    a.plot(gx + xs * gw, gy + gh - ys * gh * 0.86, color=GOLD, lw=2.2 * S,
           zorder=8)
    for fx in (0.18, 0.46, 0.84):
        fy = 1 - np.exp(-3.1 * fx)
        dot(a, gx + fx * gw, gy + gh - fy * gh * 0.86, 5.5, GOLD, z=9)
    t(a, gx + gw / 2, gy - 20, "more data, better labels", px=13, c=MUTED,
      ha="center", z=8)
    t(a, gx + gw / 2, gy + gh + 26, "each round is measured, not guessed",
      px=13, c=DIM, ha="center", z=8)


# ============================================================ 08 face match
def g08(a):
    fx, fy = 140, 180
    a.add_patch(Ellipse((fx, fy), 128, 164, fc="#20262E", ec=EDGE,
                        lw=1.5 * S, zorder=6))
    rng = np.random.RandomState(9)
    pts = []
    for k in range(26):
        ang = 2 * np.pi * k / 26
        rr = 0.78 + 0.16 * rng.rand()
        pts.append((fx + 64 * rr * np.cos(ang), fy + 82 * rr * np.sin(ang)))
    inner = [(fx - 26, fy - 22), (fx + 26, fy - 22), (fx, fy + 4),
             (fx - 22, fy + 36), (fx + 22, fy + 36), (fx, fy + 52)]
    for (x0, y0) in inner:
        for (x1, y1) in pts[::3]:
            ln(a, x0, y0, x1, y1, c=BLUE, lw=0.7, z=7, alpha=0.30)
    for (x, y) in pts + inner:
        dot(a, x, y, 3.4, BLUE, z=9)

    arr(a, 222, fy, 272, fy, lw=1.7)
    rng2 = np.random.RandomState(4)
    bx0, bw_ = 288, 11
    for i in range(14):
        hgt = 26 + 72 * rng2.rand()
        rect(a, bx0 + i * (bw_ + 5), fy + 44 - hgt, bw_, hgt, fc=GOLD,
             ec="none", z=8)
    ln(a, bx0 - 6, fy + 44, bx0 + 14 * (bw_ + 5), fy + 44, c=EDGE, lw=1.2,
       z=7)
    t(a, bx0 + 110, fy + 72, "a face becomes numbers", px=14, c=MUTED,
      ha="center", z=9)

    arr(a, 520, fy, 556, fy, lw=1.7)
    dot(a, 586, fy, 26, "#1B2A1E", z=8, ec=GREEN, lw=1.8)
    a.plot([576, 583, 597], [fy, fy + 8, fy - 10], color=GREEN, lw=2.6 * S,
           zorder=9, solid_capstyle="round")


# =============================================================== 09 optimise
def g09(a):
    specs = [("FP32", 150, 150, WHITE), ("FP16", 108, 108, GOLD),
             ("INT8", 74, 74, GREEN)]
    x = 104
    for i, (lab, w_, h_, c) in enumerate(specs):
        y = 196 - h_ / 2
        bx(a, x, y, w_, h_, fc=PANEL, ec=c, lw=1.7, r=10, z=6)
        t(a, x + w_ / 2, y + h_ / 2, lab, px=17, c=c, ha="center", w="bold",
          z=8, fam=MONO)
        t(a, x + w_ / 2, 300, ("32 bits", "16 bits", "8 bits")[i], px=13,
          c=MUTED, ha="center", z=8, fam=MONO)
        if i < 2:
            arr(a, x + w_ + 14, 196, x + w_ + 52, 196, lw=1.7)
        x += w_ + 66
    t(a, 330, 72, "the same weights, stored in fewer bits", px=16, c=GOLD,
      ha="center", z=9)
    t(a, 330, 342, "what it costs in accuracy is measured on your data",
      px=13, c=DIM, ha="center", z=9)


# ================================================================= 10 docker
def g10(a):
    ix, iy, iw, ih = 92, 86, 212, 196
    bx(a, ix, iy, iw, ih, fc="#13181E", ec=GOLD_D, lw=1.5, r=10, z=5)
    t(a, ix + iw / 2, iy + 26, "image", px=15, c=GOLD, ha="center", w="bold",
      z=8, fam=MONO)
    for i, lab in enumerate(("your code", "python deps", "base os")):
        bx(a, ix + 20, iy + 50 + i * 46, iw - 40, 38, fc=PANEL, ec=EDGE,
           lw=1.2, r=6, z=6)
        t(a, ix + iw / 2, iy + 69 + i * 46, lab, px=13, c=MUTED, ha="center",
          z=8, fam=MONO)

    arr(a, ix + iw + 16, 184, ix + iw + 72, 184, lw=1.7)
    t(a, ix + iw + 44, 160, "run", px=13, c=GOLD, ha="center", z=8, fam=MONO)

    for k, (ox, oy) in enumerate(((396, 112), (414, 130), (432, 148))):
        bx(a, ox, oy, 150, 120, fc="#13181E" if k < 2 else PANEL, ec=EDGE,
           lw=1.3, r=9, z=5 + k)
    dot(a, 452, 174, 6, GREEN, z=9)
    t(a, 468, 174, "running", px=13, c=GREEN, z=9, fam=MONO)
    t(a, 507, 214, "same on every", px=12, c=MUTED, ha="center", z=9)
    t(a, 507, 236, "machine", px=12, c=MUTED, ha="center", z=9)
    t(a, 330, 326, "one image, identical on laptop and on the board",
      px=14, c=MUTED, ha="center", z=9)


# ================================================================== 11 board
def g11(a):
    bxp, byp, bwp, bhp = 92, 92, 266, 184
    bx(a, bxp, byp, bwp, bhp, fc="#15301F", ec="#2E6B41", lw=1.6, r=8, z=5)
    for i in range(20):                                   # GPIO header
        rect(a, bxp + 22 + i * 10, byp + 14, 5, 11, fc=GOLD, ec="none", z=7)
        rect(a, bxp + 22 + i * 10, byp + 29, 5, 11, fc=GOLD, ec="none", z=7)
    bx(a, bxp + 72, byp + 72, 92, 76, fc="#1A1F26", ec="#3A434F", lw=1.4,
       r=4, z=7)
    t(a, bxp + 118, byp + 110, "SoC", px=15, c=MUTED, ha="center", w="bold",
      z=8, fam=MONO)
    for i in range(10):                                   # chip legs
        ln(a, bxp + 72, byp + 80 + i * 7, bxp + 64, byp + 80 + i * 7,
           c="#6C7480", lw=1.0, z=7)
        ln(a, bxp + 164, byp + 80 + i * 7, bxp + 172, byp + 80 + i * 7,
           c="#6C7480", lw=1.0, z=7)
    bx(a, bxp + 16, byp + 112, 40, 44, fc="#1A1F26", ec="#3A434F", lw=1.2,
       r=3, z=7)
    bx(a, bxp + 196, byp + 70, 52, 34, fc="#1A1F26", ec="#3A434F", lw=1.2,
       r=3, z=7)
    dot(a, bxp + 232, byp + 150, 6, GREEN, z=8)
    rect(a, bxp + 188, byp + 150, 26, 7, fc="#101418", ec="#3A434F", lw=1.0,
       z=7)

    # ribbon up to a camera module
    cxm, cym = 468, 118
    for k in range(9):
        ln(a, bxp + 196 + k * 2.2, byp + 70, cxm - 30 + k * 2.2, cym + 52,
           c="#8C7A5A", lw=1.1, z=6)
    bx(a, cxm - 42, cym - 30, 84, 84, fc="#1A1F26", ec=EDGE, lw=1.4, r=8, z=7)
    dot(a, cxm, cym + 12, 22, "#0B0E12", z=8, ec=GOLD_D, lw=1.5)
    dot(a, cxm, cym + 12, 9, "#05070A", z=9, ec=GOLD, lw=1.2)
    t(a, cxm, cym + 76, "camera", px=13, c=MUTED, ha="center", z=9, fam=MONO)

    chip(a, 400, 216, "no laptop")
    chip(a, 400, 254, "no internet")
    t(a, 330, 330, "the model now runs on the board itself", px=15, c=MUTED,
      ha="center", z=9)


# ==================================================================== 12 iot
def g12(a):
    bx(a, 254, 148, 152, 84, fc=PANEL, ec=GOLD_D, lw=1.6, r=10, z=7)
    t(a, 330, 178, "MQTT", px=18, c=GOLD, ha="center", w="bold", z=9,
      fam=MONO)
    t(a, 330, 206, "broker", px=13, c=MUTED, ha="center", z=9, fam=MONO)

    # publisher
    bx(a, 76, 152, 118, 76, fc="#13181E", ec=EDGE, lw=1.4, r=9, z=6)
    t(a, 135, 178, "edge AI", px=14, c=WHITE, ha="center", z=8, fam=MONO)
    t(a, 135, 202, "result", px=12, c=MUTED, ha="center", z=8, fam=MONO)
    arr(a, 200, 190, 248, 190, lw=1.6)
    t(a, 224, 164, "publish", px=11, c=GOLD, ha="center", z=8, fam=MONO)

    subs = [("dashboard", 96), ("database", 190), ("alert", 284)]
    for lab, yy in subs:
        bx(a, 468, yy - 26, 132, 56, fc="#13181E", ec=EDGE, lw=1.3, r=9, z=6)
        t(a, 534, yy + 2, lab, px=13, c=WHITE, ha="center", z=8, fam=MONO)
        ln(a, 412, 190, 440, yy, c=GOLD_D, lw=1.4, z=6)
        arr(a, 440, yy, 462, yy, c=GOLD_D, lw=1.4)
    t(a, 330, 266, "tesr/line1/count", px=13, c=MUTED, ha="center", z=9,
      fam=MONO)
    t(a, 330, 334, "a detection is only useful once something acts on it",
      px=14, c=MUTED, ha="center", z=9)


# =========================================================== 13 end to end
def g13(a):
    stages = [("camera", 86), ("model", 218), ("broker", 350), ("screen", 482)]
    for lab, x in stages:
        bx(a, x, 150, 92, 76, fc=PANEL, ec=GOLD_D, lw=1.5, r=10, z=7)
        t(a, x + 46, 188, lab, px=14, c=GOLD, ha="center", w="bold", z=9,
          fam=MONO)
    for i in range(3):
        arr(a, stages[i][1] + 97, 188, stages[i + 1][1] - 5, 188, lw=1.7)
    for x, lab in ((86, "see"), (218, "decide"), (350, "send"),
                   (482, "show")):
        t(a, x + 46, 250, lab, px=14, c=MUTED, ha="center", z=9)
    t(a, 330, 92, "every piece of the two days, in one line", px=17, c=WHITE,
      ha="center", z=9)
    t(a, 330, 320, "you build it, you run it, you show it working", px=14,
      c=MUTED, ha="center", z=9)


# ====================================================== 14 track and decode
def g14(a):
    sx, sy, sw, sh = 76, 76, 310, 212
    bx(a, sx, sy, sw, sh, fc="#0C1014", ec=EDGE, lw=1.3, r=10, z=5)
    ln(a, sx + 158, sy + 14, sx + 158, sy + sh - 14, c=GOLD, lw=2.2, z=7,
       ls=(0, (7, 5)))
    t(a, sx + 158, sy + sh - 26, "count line", px=12, c=GOLD, ha="center",
      z=9, fam=MONO)
    for dx, idn, col in ((34, "id 7", GREEN), (196, "id 9", GREEN)):
        px_, py_ = sx + dx + 34, sy + 108
        dot(a, px_, py_ - 40, 12, STEEL, z=6)
        a.add_patch(Polygon([(px_ - 15, py_ + 36), (px_ - 10, py_ - 26),
                             (px_ + 10, py_ - 26), (px_ + 15, py_ + 36)],
                            closed=True, fc=STEEL, ec="none", zorder=6))
        rect(a, px_ - 27, py_ - 58, 54, 100, ec=col, lw=1.8, z=8)
        tag(a, px_ - 27, py_ - 58, idn, c=col, px=10)
    arr(a, sx + 96, sy + 44, sx + 140, sy + 44, c=GREEN, lw=1.5, head=11)
    t(a, sx + 118, sy + 28, "in", px=12, c=GREEN, ha="center", z=9, fam=MONO)

    # the codes it can read
    try:
        import cv2
        dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        mk = cv2.aruco.generateImageMarker(dic, 7, 240)
    except Exception:
        mk = np.full((240, 240), 255, np.uint8)
    try:
        import io
        import segno
        buf = io.BytesIO()
        segno.make_qr("TESR", error="m").save(buf, kind="png", scale=8,
                                              border=2)
        buf.seek(0)
        qr = np.array(Image.open(buf).convert("L"))
    except Exception:
        qr = np.full((240, 240), 255, np.uint8)

    bx(a, 414, 76, 190, 212, fc=PANEL2, ec=EDGE, lw=1.3, r=10, z=5)
    rect(a, 436, 100, 68, 68, fc="#E6E3DE", ec="none", z=7)
    img(a, qr, 441, 105, 499, 163, z=8)
    rect(a, 518, 100, 68, 68, fc="#E6E3DE", ec="none", z=7)
    img(a, mk, 523, 105, 581, 163, z=8)
    t(a, 470, 186, "QR", px=13, c=MUTED, ha="center", z=9, fam=MONO)
    t(a, 552, 186, "ArUco", px=13, c=MUTED, ha="center", z=9, fam=MONO)
    rect(a, 436, 208, 150, 44, fc="#E6E3DE", ec="none", z=7)
    t(a, 511, 230, "PART A-2140", px=14, c="#1A1A1C", ha="center", z=8,
      fam=MONO)
    t(a, 511, 272, "text", px=13, c=MUTED, ha="center", z=9, fam=MONO)


# ===================================================================== build
SPEC = [
    (1, g01, "Install it once, then prove it runs."),
    (2, g02, "The short version — page 2 has all eight stages."),
    (3, g03, "To a computer, a picture is a grid of numbers."),
    (4, g04, "Layers, a loss, and a few epochs."),
    (5, g05, "Find the object. Draw the box. Name it."),
    (6, g06, "Your own images become your own model."),
    (7, g07, "Measure what is wrong before changing anything."),
    (8, g08, "A face becomes a vector you can compare."),
    (9, g09, "The same model, stored in fewer bits."),
    (10, g10, "Pack the whole environment, not just the code."),
    (11, g11, "Off the laptop and onto the board."),
    (12, g12, "Publish the result so something can act on it."),
    (13, g13, "Camera to dashboard, built by you, end to end."),
    (14, g14, "Track it, count it, read it, locate it."),
]

if __name__ == "__main__":
    tiles = []
    for n, fn, cap in SPEC:
        f, a = plate(cap)
        fn(a)
        path = OUT + "ch%02d_cover.png" % n
        f.savefig(path, dpi=100, facecolor=BG)
        plt.close(f)
        im = Image.open(path)
        print("ch%02d_cover  %s  %s" % (n, im.size, cap))
        tiles.append(im)

    # contact sheet, so every glyph can be checked side by side
    cols, rows = 3, 5
    tw, th = 440, 313
    sheet = Image.new("RGB", (cols * tw, rows * th), (8, 10, 13))
    for i, im in enumerate(tiles):
        sheet.paste(im.resize((tw - 8, th - 8), Image.LANCZOS),
                    ((i % cols) * tw + 4, (i // cols) * th + 4))
    sheet.save(OUT + "ch_cover_sheet.png")
    print("sheet", sheet.size)
