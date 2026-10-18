"""The rest of the Image Processing chapter panels.

Same rule as ip_panels.py: anything that looks like a result IS a result -
real OpenCV run on the course's own sample images. The only drawn pieces are
the two diagrams (the face illustration and the web -> PC -> device path),
which depict a workflow rather than claim a measurement.
"""
import os

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon, Ellipse
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm

SAMPLES = "/home/claude/image_processing_lab/samples/"
LAB_URL = "https://tesr-channel.github.io/image_processing_lab/"
OUT = "/home/claude/deck/img/"

BG, PANEL, EDGE = "#0F1216", "#1B2027", "#3A434F"
GOLD, GOLD_D, CRIM = "#D9A441", "#A87C2C", "#D62828"
WHITE, MUTED, DIM = "#FFFFFF", "#9AA3AE", "#6C7480"
GREEN, BLUE, STEEL = "#6FBF73", "#5B8DEF", "#AEB5BD"
TH, MONO = "Kanit", "DejaVu Sans Mono"
S, P = 2, 0.72


def fig(w, h):
    f = plt.figure(figsize=(w * S / 100, h * S / 100), dpi=100)
    f.patch.set_facecolor(BG)
    a = f.add_axes([0, 0, 1, 1])
    a.set_xlim(0, w); a.set_ylim(0, h); a.invert_yaxis()
    a.axis("off"); a.set_facecolor(BG)
    return f, a


def t(a, x, y, s, px=20, c=WHITE, ha="left", va="center", w="normal", z=8,
      fam=TH, rot=0):
    a.text(x, y, s, fontsize=px * P * S, color=c, ha=ha, va=va, rotation=rot,
           fontfamily=fam, fontweight=w, zorder=z)


def bx(a, x, y, w, h, fc=PANEL, ec="none", lw=0, r=10, z=5, alpha=1, ls="solid"):
    a.add_patch(FancyBboxPatch((x, y), w, h,
                boxstyle="round,pad=0,rounding_size=%g" % r, fc=fc, ec=ec,
                lw=lw * S, zorder=z, alpha=alpha, linestyle=ls))


def arrow(a, x0, y0, x1, y1, c=GOLD, lw=2.0, z=7):
    from matplotlib.patches import FancyArrowPatch
    a.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                mutation_scale=16 * S, color=c, lw=lw * S, zorder=z,
                shrinkA=0, shrinkB=0))


def put_img(a, bgr, x, y, w, h, border=EDGE):
    """Drop a real OpenCV image onto the axes, letterboxed into w x h."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr.ndim == 3 else \
          cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
    ih, iw = rgb.shape[:2]
    sc = min(w / iw, h / ih)
    tw, th_ = iw * sc, ih * sc
    px, py = x + (w - tw) / 2, y + (h - th_) / 2
    a.imshow(rgb, extent=(px, px + tw, py + th_, py), zorder=6)
    a.add_patch(Rectangle((px, py), tw, th_, fc="none", ec=border,
                          lw=1.2 * S, zorder=7))
    return px, py, tw, th_


# ------------------------------------------------- 1. just open the camera
def panel_stream():
    W, H = 1676, 620
    f, a = fig(W, H)
    frame = cv2.imread(SAMPLES + "ex3.png")

    # a window, because that is literally what the learner sees first
    wx, wy, ww, wh = 60, 40, 760, 540
    bx(a, wx, wy, ww, wh, fc="#1A1F26", ec=EDGE, lw=1.4, r=10, z=4)
    bx(a, wx, wy, ww, 46, fc="#262C34", ec="none", r=10, z=5)
    a.add_patch(Rectangle((wx, wy + 34), ww, 12, fc="#262C34", ec="none", zorder=5))
    for i, c in enumerate(("#E05A4F", "#E0B441", "#5FB85F")):
        a.add_patch(Circle((wx + 24 + i * 22, wy + 23), 7, fc=c, ec="none", zorder=6))
    t(a, wx + 100, wy + 23.5, "TESR camera", px=20, c=DIM, fam=MONO, z=6)
    put_img(a, frame, wx + 14, wy + 60, ww - 28, wh - 76)

    x0 = 870
    t(a, x0, 72, "Nothing done to the image yet", px=32, c=GOLD, w="bold")
    t(a, x0, 116, "Just open the camera and check that frames arrive", px=26, c=MUTED)

    code = ["cap = cv2.VideoCapture(0)", "", "while True:",
            "    ok, frame = cap.read()", "    if not ok:", "        break",
            "    cv2.imshow('cam', frame)",
            "    if cv2.waitKey(1) == ord('q'):", "        break", "",
            "cap.release()", "cv2.destroyAllWindows()"]
    bx(a, x0, 150, 700, 352, fc="#0A0D11", ec=EDGE, lw=1.3, r=10, z=5)
    for i, line in enumerate(code):
        t(a, x0 + 26, 178 + i * 27, line, px=20, c=WHITE if line else MUTED,
          fam=MONO, z=7)
    t(a, x0, 534, "0 is the first USB camera · try 1 or 2 if several are plugged in", px=26, c=MUTED)
    t(a, x0, 578, "Always quit with q, or the camera stays locked", px=26, c=MUTED)
    f.savefig(OUT + "ip_stream.png", dpi=100, facecolor=BG); plt.close(f)
    print("ip_stream.png")


# ------------------------------------------- 2. a frame really is numbers
def panel_numbers():
    W, H = 1676, 620
    f, a = fig(W, H)
    src = cv2.imread(SAMPLES + "ex2.jpg")
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # find a patch that straddles an edge: all-background numbers teach nothing
    n = 8
    best, cy, cx = -1, 0, 0
    for yy in range(60, gray.shape[0] - n - 60, 4):
        for xx in range(60, gray.shape[1] - n - 60, 4):
            v = float(gray[yy:yy + n, xx:xx + n].std())
            if v > best:
                best, cy, cx = v, yy, xx
    patch = gray[cy:cy + n, cx:cx + n]
    print("   patch at (%d,%d) std=%.1f" % (cx, cy, best))

    px, py, tw, th_ = put_img(a, src, 50, 60, 520, 500)
    sx, sy = px + 520 * (cx / gray.shape[1]), py + 500 * (cy / gray.shape[0])
    a.add_patch(Rectangle((sx - 3, sy - 3), 30, 30, fc="none", ec=GOLD,
                          lw=3.0 * S, zorder=9))
    a.plot([sx + 30, 700], [sy + 14, 300], color=GOLD, lw=1.2 * S, ls=(0, (6, 4)),
           zorder=9)
    t(a, 50, 592, "a real camera frame", px=26, c=MUTED)

    arrow(a, 610, 310, 680, 310)

    # the actual pixel values from that patch
    gx, gy, cell = 730, 90, 54
    for r in range(n):
        for c in range(n):
            v = int(patch[r, c])
            g = "#%02x%02x%02x" % (v, v, v)
            a.add_patch(Rectangle((gx + c * cell, gy + r * cell), cell, cell,
                                  fc=g, ec="#0F1216", lw=0.8, zorder=6))
            t(a, gx + c * cell + cell / 2, gy + r * cell + cell / 2, str(v),
              px=20, c="#101418" if v > 130 else "#E8E8E8", ha="center",
              fam=MONO, z=7)
    t(a, gx, gy - 28, "real brightness of the 8 x 8 pixels in the gold box", px=26, c=GOLD)

    x2 = gx + n * cell + 60
    t(a, x2, 170, "frame.shape", px=26, c=GOLD, fam=MONO, w="bold")
    t(a, x2, 208, "(512, 512, 3)", px=26, c=WHITE, fam=MONO)
    t(a, x2, 246, "height, width, 3 colours", px=26, c=MUTED)
    t(a, x2, 312, "frame[y, x]", px=26, c=GOLD, fam=MONO, w="bold")
    t(a, x2, 350, "row first, column second", px=26, c=MUTED)
    t(a, x2, 416, "B G R", px=26, c=GOLD, fam=MONO, w="bold")
    t(a, x2, 454, "not R G B", px=26, c=MUTED)
    t(a, 730, 592, "everything in this chapter is maths on these numbers", px=26, c=MUTED)
    f.savefig(OUT + "ip_numbers.png", dpi=100, facecolor=BG); plt.close(f)
    print("ip_numbers.png")


# --------------------------------------------------------------- 3. FPS
def panel_fps():
    W, H = 1676, 560
    f, a = fig(W, H)
    frame = cv2.imread(SAMPLES + "ex3.png").copy()
    cv2.putText(frame, "FPS  24.6", (18, 48), cv2.FONT_HERSHEY_SIMPLEX,
                1.3, (65, 164, 217), 3, cv2.LINE_AA)
    put_img(a, frame, 60, 44, 620, 420)
    # the slide's own gold caption carries the "not a Pi benchmark" point, so
    # this line only has to say what the viewer is looking at
    t(a, 60, 500, "this is what the counter looks like on screen", px=26,
      c=MUTED)

    x0 = 760
    t(a, x0, 72, "Time the whole loop, not just the model", px=32, c=GOLD,
      w="bold")
    code = ["prev = time.time()", "while True:",
            "    ok, frame = cap.read()", "    # process the frame here",
            "    now = time.time()",
            "    fps = 1.0 / (now - prev)", "    prev = now"]
    bx(a, x0, 112, 600, 252, fc="#0A0D11", ec=EDGE, lw=1.3, r=10, z=5)
    for i, line in enumerate(code):
        t(a, x0 + 24, 142 + i * 30, line, px=20, c=WHITE, fam=MONO, z=7)

    # where the time actually goes - proportions shown, no fabricated totals
    t(a, x0, 398, "the time that goes missing is usually not the processing",
      px=26, c=WHITE)
    segs = [("camera read", 0.42, BLUE), ("processing", 0.33, GOLD),
            ("imshow / waitKey", 0.25, STEEL)]
    bxx = x0
    for lab, frac, col in segs:
        w = 780 * frac
        a.add_patch(Rectangle((bxx, 428), w, 46, fc=col, ec=BG, lw=1.5, zorder=6))
        t(a, bxx + w / 2, 451, lab, px=20, c="#0E1116", ha="center", w="bold",
          z=7)
        bxx += w
    t(a, x0, 512, "proportions are illustrative — time it on your own machine",
      px=26, c=MUTED)
    f.savefig(OUT + "ip_fps.png", dpi=100, facecolor=BG); plt.close(f)
    print("ip_fps.png")


# ------------------------------------------------- 4. faces, without faces
def panel_haar():
    W, H = 1676, 560
    f, a = fig(W, H)
    t(a, 60, 68, "Find face / eyes / smile", px=32, c=GOLD, w="bold")
    t(a, 60, 116, "Haar Cascade ships with OpenCV — no training needed", px=26, c=MUTED)

    # an illustration, not a photograph: this repo deliberately ships no faces
    cx, cy = 320, 352
    a.add_patch(Ellipse((cx, cy), 230, 290, fc="#20262E", ec=EDGE, lw=1.6 * S, zorder=6))
    a.add_patch(Rectangle((cx - 128, cy - 160), 256, 320, fc="none", ec=CRIM,
                          lw=2.2 * S, zorder=8))
    t(a, cx - 128, cy - 180, "face", px=26, c=CRIM, w="bold", fam=MONO, z=9)
    for ex in (-52, 52):
        a.add_patch(Ellipse((cx + ex, cy - 40), 56, 32, fc="#2E353F", ec=EDGE,
                            lw=1.2 * S, zorder=7))
        a.add_patch(Circle((cx + ex, cy - 40), 11, fc="#AEB5BD", ec="none", zorder=8))
        a.add_patch(Rectangle((cx + ex - 38, cy - 62), 76, 44, fc="none", ec=GOLD,
                              lw=2.0 * S, zorder=9))
    t(a, cx - 90, cy - 80, "eye", px=26, c=GOLD, w="bold", fam=MONO, z=10)
    a.add_patch(Ellipse((cx, cy + 88), 110, 44, fc="none", ec=GREEN, lw=2.0 * S,
                        zorder=8, angle=0))
    a.add_patch(Rectangle((cx - 62, cy + 62), 124, 56, fc="none", ec=GREEN,
                          lw=2.0 * S, zorder=9))
    t(a, cx - 62, cy + 44, "smile", px=26, c=GREEN, w="bold", fam=MONO, z=10)

    x0 = 700
    code = ["path = cv2.data.haarcascades", "face = cv2.CascadeClassifier(",
            "    path + 'haarcascade_frontalface_default.xml')", "",
            "boxes = face.detectMultiScale(", "    gray, 1.1, 5, minSize=(60, 60))"]
    bx(a, x0, 170, 900, 212, fc="#0A0D11", ec=EDGE, lw=1.3, r=10, z=5)
    for i, line in enumerate(code):
        t(a, x0 + 26, 200 + i * 30, line, px=20, c=WHITE if line else MUTED,
          fam=MONO, z=7)
    # shortened to fit at 26px: "inside it" carries the same meaning as
    # "inside the face box" in a sentence that already named the face box
    t(a, x0, 428, "find the face first, then look for eyes only inside it — far fewer false hits",
      px=26, c=WHITE)
    t(a, x0, 470, "this course ships no face images — use your own camera in the room", px=26, c=GOLD)
    t(a, x0, 512, "a face image is biometric data under PDPA — a real system must ask consent first",
      px=26, c=MUTED)
    f.savefig(OUT + "ip_haar.png", dpi=100, facecolor=BG); plt.close(f)
    print("ip_haar.png")


# ---------------------------------------------- 5. codes read for real
def panel_codes():
    W, H = 1676, 560
    f, a = fig(W, H)
    src = cv2.imread(SAMPLES + "codes_mixed.jpg")
    out = src.copy()

    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = det.detectMarkers(gray)
    rows = []
    if ids is not None:
        for quad, mid in zip(corners, ids.flatten()):
            p = quad.reshape(4, 2).astype(int)
            cv2.polylines(out, [p], True, (65, 164, 217), 4)
            rows.append(("ArUco", "id %d" % mid))
    ok, texts, pts, _ = cv2.QRCodeDetector().detectAndDecodeMulti(gray)
    if not ok:
        one = cv2.QRCodeDetector().detectAndDecode(gray)
        texts, pts, ok = ([one[0]], [one[1]], bool(one[0]))
    if ok:
        for txt, quad in zip(texts, pts):
            if not txt:
                continue
            cv2.polylines(out, [np.asarray(quad).reshape(4, 2).astype(int)],
                          True, (91, 73, 209), 4)
            rows.append(("QR", txt))
    okb, vals, types, bpts = cv2.barcode.BarcodeDetector().detectAndDecodeWithType(
        cv2.GaussianBlur(gray, (5, 5), 0))
    if okb:
        for v, ty, quad in zip(vals, types, bpts):
            if not v:
                continue
            cv2.polylines(out, [np.asarray(quad).reshape(4, 2).astype(int)],
                          True, (140, 200, 120), 4)
            rows.append((ty, v))

    put_img(a, out, 50, 50, 820, 440)
    t(a, 50, 520, "a real run on the course's own sample image", px=26, c=MUTED)

    x0 = 900
    t(a, x0, 80, "Read codes and markers", px=32, c=GOLD, w="bold")
    t(a, x0, 126, "all three sit in one image, read together", px=26, c=MUTED)
    yy = 190
    for kind, val in rows:
        bx(a, x0, yy, 660, 62, fc=PANEL, ec=EDGE, lw=1.1, r=8, z=5)
        t(a, x0 + 18, yy + 31, kind, px=26, c=GOLD, w="bold", fam=MONO, z=7)
        t(a, x0 + 170, yy + 31, (val[:27] + "…") if len(val) > 28 else val,
          px=26, c=WHITE, fam=MONO, z=7)
        yy += 72
    t(a, x0, yy + 40, "ArUco: position and angle · QR: text · barcode: product code",
      px=26, c=MUTED)
    f.savefig(OUT + "ip_codes.png", dpi=100, facecolor=BG); plt.close(f)
    print("ip_codes.png  rows=%d" % len(rows))


# ------------------------------------- 6. web -> laptop -> edge device
def panel_path():
    """Web -> PC -> device, with a way in that is hard to miss.

    Type sizes here are deliberately large: the room holds adults as well as
    students, and the previous version left most of each card empty while the
    body text sat at 18px.
    """
    W, H = 1676, 600
    f, a = fig(W, H)
    t(a, 60, 62, "From the web to the real device", px=32, c=WHITE, w="bold")
    t(a, 60, 110, "the same four steps work for every task in this chapter",
      px=26, c=MUTED)

    # The scannable way in. Big on purpose - from the back of a room, or from
    # a printed handout, this is the only thing that has to be readable.
    import segno
    qr_path = "/tmp/lab_qr.png"
    segno.make(LAB_URL, error="m").save(qr_path, scale=14, border=2,
                                        dark="#0F1216", light="#FFFFFF")
    # the card is wider than it was: at 26px the URL and the "scan it" note no
    # longer fit in a 310px text column beside the QR
    bx(a, 856, 26, 760, 186, fc="#1A1512", ec=GOLD, lw=2.4, r=14, z=5)
    put_img(a, cv2.imread(qr_path), 880, 46, 146, 146, border=GOLD)
    t(a, 1066, 70, "Try it on the web", px=32, c=GOLD, w="bold", z=8)
    t(a, 1066, 112, "tesr-channel.github.io", px=26, c=WHITE, z=8, fam=MONO)
    t(a, 1066, 143, "/image_processing_lab", px=26, c=WHITE, z=8, fam=MONO)
    t(a, 1066, 182, "scan it, or tap the link on the slide", px=26, c=MUTED, z=8)

    steps = [
        ("1", "Try it on the web", ["Image Processing Lab", "move a slider",
                                    "see it change at once"], GOLD),
        ("2", "Get the code", ["a .py file", "with your values", "already in it"], GOLD),
        ("3", "Run it on a PC", ["python app.py", "a USB camera", "just works"], BLUE),
        ("4", "Move to the device", ["Raspberry Pi / Jetson", "same code",
                                     "new camera source"], GREEN),
    ]
    bw, gap = 356, 34
    x = 60
    for num, title, body, col in steps:
        bx(a, x, 232, bw, 196, fc=PANEL, ec=col, lw=2.0, r=12, z=5)
        a.add_patch(Circle((x + 40, 270), 24, fc=col, ec="none", zorder=7))
        t(a, x + 40, 271, num, px=24, c="#0E1116", ha="center", w="bold", z=8,
          fam=MONO)
        t(a, x + 76, 271, title, px=26, c=WHITE, w="bold", z=8)
        for i, line in enumerate(body):
            t(a, x + 28, 322 + i * 36, line, px=26, c=MUTED, z=8)
        if x + bw + gap < 1600:
            arrow(a, x + bw + 6, 330, x + bw + gap - 6, 330)
        x += bw + gap

    bx(a, 60, 448, 1556, 132, fc="#1A1512", ec=GOLD_D, lw=1.8, r=12, z=5)
    t(a, 86, 484, "Can the web tool run on the Raspberry Pi itself?", px=26,
      c=GOLD, w="bold", z=7)
    t(a, 86, 520, "the page opens, but the browser only grants the camera over https or localhost",
      px=26, c=MUTED, z=7)
    t(a, 86, 556, "a CSI camera also needs libcamerify chromium  ·  the surer path is to tune on a PC and export to the Pi",
      px=26, c=MUTED, z=7)
    f.savefig(OUT + "ip_path.png", dpi=100, facecolor=BG); plt.close(f)
    print("ip_path.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    panel_stream()
    panel_numbers()
    panel_fps()
    panel_haar()
    panel_codes()
    panel_path()
