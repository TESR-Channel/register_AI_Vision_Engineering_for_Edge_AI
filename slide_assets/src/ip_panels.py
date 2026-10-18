"""Before/after panels for the Image Processing chapter.

Every result here is produced by actually running OpenCV on the course's own
sample images - the same files that ship in the Image Processing Lab, so a
learner sees the same picture on the slide and in the tool. Nothing is drawn
by hand to look like a result.

Output: /home/claude/deck/img/ip_*.png, sized for a 1920x1080 slide at 2x.
"""
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm

SAMPLES = "/home/claude/image_processing_lab/samples/"
OUT = "/home/claude/deck/img/"

BG = (15, 18, 22)          # #0F1216, the deck background
PANEL = (27, 32, 39)
EDGE = (58, 67, 79)
GOLD = (217, 164, 65)
GREEN = (111, 191, 115)
MUTED = (154, 163, 174)
WHITE = (240, 240, 242)

S = 2                      # render at 2x the slide rectangle
W, H = 1676, 620           # slide px


def font(name, px):
    path = fm.findfont(fm.FontProperties(family=name), fallback_to_default=True)
    return ImageFont.truetype(path, int(px * S))


def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else \
           cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def strip(frames, labels, out_name, note=None, w=W, h=H):
    """Lay out N images left to right with gold arrows between them."""
    assert len(frames) == len(labels)
    n = len(frames)
    img = Image.new("RGB", (w * S, h * S), BG)
    d = ImageDraw.Draw(img)

    f_lab = font("Kanit", 27)
    f_note = font("Kanit", 26)

    PAD = 14
    gap = 46
    arrow = 42 if n > 1 else 0
    usable = w - 2 * PAD
    cell_w = (usable - (n - 1) * (gap + arrow)) // n
    top = 58
    cell_h = h - top - (84 if note else 40)

    x = PAD
    for i, (fr, lab) in enumerate(zip(frames, labels)):
        rgb = bgr2rgb(fr)
        ih, iw = rgb.shape[:2]
        sc = min(cell_w / iw, cell_h / ih)
        tw_, th_ = int(iw * sc), int(ih * sc)
        tile = Image.fromarray(cv2.resize(rgb, (tw_ * S, th_ * S),
                                          interpolation=cv2.INTER_AREA))
        px = (x + (cell_w - tw_) // 2) * S
        py = (top + (cell_h - th_) // 2) * S
        img.paste(tile, (px, py))
        d.rectangle([px - 2, py - 2, px + tw_ * S + 1, py + th_ * S + 1],
                    outline=EDGE, width=2)
        # the label sits above its own tile, not in a legend somewhere else.
        # anchored on the glyph-box bottom so a descender can never clip into
        # the tile, which the top-anchored version did when a tile filled the
        # cell exactly (ip_draw)
        d.text((px, (top - 6) * S), lab, font=f_lab, anchor="ld",
               fill=GOLD if i == n - 1 else MUTED)
        if i < n - 1:
            ax = (x + cell_w + gap // 2) * S
            ay = (top + cell_h // 2) * S
            d.polygon([(ax, ay - 16 * S), (ax + 30 * S, ay), (ax, ay + 16 * S)],
                      fill=GOLD)
        x += cell_w + gap + arrow

    if note:
        d.text((PAD * S, (h - 46) * S), note, font=f_note, fill=MUTED)

    img.save(OUT + out_name)
    print("%-26s %s" % (out_name, img.size))
    return img


# ---------------------------------------------------------------- lessons
def lesson_colour():
    """HSV colour pick on the cluttered wooden-table scene."""
    src = cv2.imread(SAMPLES + "ex3.png")
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    # red wraps around H=0, so it needs two bands - the single thing people
    # get wrong first
    m1 = cv2.inRange(hsv, (0, 120, 70), (4, 255, 255))
    m2 = cv2.inRange(hsv, (172, 120, 70), (179, 255, 255))
    mask = cv2.morphologyEx(cv2.bitwise_or(m1, m2), cv2.MORPH_OPEN,
                            np.ones((5, 5), np.uint8))
    out = src.copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n = 0
    for c in cnts:
        if cv2.contourArea(c) < 400:
            continue
        n += 1
        (cx, cy), r = cv2.minEnclosingCircle(c)
        cv2.circle(out, (int(cx), int(cy)), int(r), (65, 164, 217), 3)
    cv2.putText(out, "red: %d" % n, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (65, 164, 217), 2, cv2.LINE_AA)
    strip([src, mask, out],
          ["camera frame", "mask of the picked colour", "%d red discs" % n],
          "ip_colour.png",
          )
    return n


def lesson_coins_blur():
    """The lesson the whole blur idea exists for: texture inside a coin is noise."""
    src = cv2.imread(SAMPLES + "ex4.png")
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    sharp_edges = cv2.Canny(gray, 80, 160)
    blur = cv2.medianBlur(gray, 15)
    blur_edges = cv2.Canny(blur, 80, 160)

    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=60,
                               param1=160, param2=55, minRadius=30, maxRadius=90)
    out = src.copy()
    n = 0
    if circles is not None:
        for x, y, r in np.uint16(np.around(circles[0])):
            n += 1
            cv2.circle(out, (x, y), r, (65, 164, 217), 3)
            cv2.circle(out, (x, y), 3, (91, 73, 209), -1)
    cv2.putText(out, "coins: %d" % n, (12, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, (65, 164, 217), 3, cv2.LINE_AA)
    strip([sharp_edges, blur_edges, out],
          ["Canny on raw — texture is noise", "medianBlur 15, then Canny",
           "%d coins found" % n],
          "ip_coins.png",
          )
    return n


def lesson_lines():
    """Hough lines on a real factory floor, with the angle filter that makes it usable."""
    src = cv2.imread(SAMPLES + "452659.jpg")
    h0, w0 = src.shape[:2]
    src = src[int(h0 * 0.18):int(h0 * 0.72), :]          # a landscape band of floor
    src = cv2.resize(src, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 160)

    raw = src.copy()
    segs = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=90, maxLineGap=18)
    kept = src.copy()
    nk = 0
    if segs is not None:
        for x1, y1, x2, y2 in segs.reshape(-1, 4):
            cv2.line(raw, (x1, y1), (x2, y2), (65, 164, 217), 5)
            ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            ang = ang if ang <= 90 else 180 - ang
            if ang >= 60:                      # keep the near-vertical ones only
                nk += 1
                cv2.line(kept, (x1, y1), (x2, y2), (111, 191, 115), 6)
    cv2.putText(kept, "kept: %d" % nk, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (111, 191, 115), 2, cv2.LINE_AA)
    strip([edges, raw, kept],
          ["Canny edges", "every line Hough found", "%d lines kept, 60–90° only" % nk],
          "ip_lines.png",
          )
    return nk


def lesson_shape_angle():
    """Rectangles and how far each one is rotated - minAreaRect gives the angle."""
    src = cv2.imread(SAMPLES + "codes_mixed.jpg")
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # the cards are near-white on a grey surface, so a fixed cut separates them
    th = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)[1]
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = src.copy()
    found = 0
    for c in cnts:
        if cv2.contourArea(c) < 20000:
            continue
        rect = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), ang = rect
        if min(rw, rh) < 60:
            continue
        found += 1
        box = np.int32(cv2.boxPoints(rect))
        cv2.drawContours(out, [box], 0, (111, 191, 115), 4)
        # OpenCV's angle convention moves between versions, so fold it into
        # (-45, 45] - the tilt a person would actually describe
        ang = ((ang + 45) % 90) - 45
        txt = "%+.1f deg" % ang
        org = (int(cx) - 95, int(cy) + 10)
        cv2.putText(out, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 24), 7, cv2.LINE_AA)
        cv2.putText(out, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 1.1, (65, 164, 217), 3, cv2.LINE_AA)
    strip([src, th, out],
          ["camera frame", "objects split from the surface", "tilt angle of each piece"],
          "ip_angle.png",
          )
    return found


def lesson_draw():
    """One call per panel, so each result can be traced to the line above it.

    The earlier version drew all four calls onto one frame, which made it
    impossible to tell which call produced which mark.
    """
    src = cv2.imread(SAMPLES + "ex2.jpg")
    BLUE_ = (217, 164, 65)
    GRN_ = (111, 191, 115)
    VIO_ = (91, 73, 209)

    def only(fn):
        o = src.copy()
        fn(o)
        return o

    items = [
        ("cv2.line(img, (30,30), (480,30), col, 4)",
         "a straight line from one point to another",
         only(lambda o: cv2.line(o, (30, 30), (480, 30), BLUE_, 6))),
        ("cv2.rectangle(img, (60,190), (250,380), col, 4)",
         "a box - top-left corner to bottom-right",
         only(lambda o: cv2.rectangle(o, (60, 190), (250, 380), GRN_, 6))),
        ("cv2.circle(img, (330,300), 70, col, 4)",
         "a circle - one centre point and a radius",
         only(lambda o: cv2.circle(o, (330, 300), 70, VIO_, 6))),
        ('cv2.putText(img, "TESR", (60,170), FONT, 1.4, col, 4)',
         "text - Latin only, cv2 cannot draw Thai",
         only(lambda o: cv2.putText(o, "TESR", (60, 170),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                                    (30, 30, 34), 5, cv2.LINE_AA))),
    ]

    img = Image.new("RGB", (W * S, H * S), BG)
    d = ImageDraw.Draw(img)
    f_code = font("DejaVu Sans Mono", 22)
    f_eff = font("Kanit", 26)
    f_note = font("Kanit", 26)

    # the grid moves up: at 26px the footnote's descenders reached the canvas
    # edge where it used to sit
    PAD, gx, gy = 14, 24, 12
    cw = (W - 2 * PAD - gx) // 2
    top, chh = 24, 262
    side = 204

    for i, (code, effect, frame) in enumerate(items):
        cx = PAD + (i % 2) * (cw + gx)
        cy = top + (i // 2) * (chh + gy)
        d.rounded_rectangle([cx * S, cy * S, (cx + cw) * S, (cy + chh) * S],
                            radius=10 * S, fill=PANEL, outline=EDGE, width=2)
        d.text(((cx + 18) * S, (cy + 14) * S), code, font=f_code, fill=GOLD)

        rgbf = bgr2rgb(frame)
        ih, iw = rgbf.shape[:2]
        sc = side / max(ih, iw)
        tw_, th_ = int(iw * sc), int(ih * sc)
        tile = Image.fromarray(cv2.resize(rgbf, (tw_ * S, th_ * S),
                                          interpolation=cv2.INTER_AREA))
        px = (cx + 18) * S
        py = (cy + 52) * S
        img.paste(tile, (px, py))
        d.rectangle([px - 2, py - 2, px + tw_ * S + 1, py + th_ * S + 1],
                    outline=EDGE, width=2)

        tx = cx + 18 + tw_ + 26
        maxw = (cx + cw - 18 - tx) * S
        words, lines, cur = effect.split(), [], ""
        for wd in words:
            cand = (cur + " " + wd).strip()
            if d.textlength(cand, font=f_eff) <= maxw or not cur:
                cur = cand
            else:
                lines.append(cur)
                cur = wd
        if cur:
            lines.append(cur)
        ty = cy + 52 + max(0, (th_ - len(lines) * 36) // 2)
        for ln in lines:
            d.text((tx * S, ty * S), ln, font=f_eff, fill=WHITE)
            ty += 36

    d.text((PAD * S, (H - 44) * S),
           "(x, y) counts from the top-left corner  ·  colours are BGR, not "
           "RGB  ·  thickness -1 fills the shape instead of outlining it",
           font=f_note, fill=MUTED)
    img.save(OUT + "ip_draw.png")
    print("ip_draw.png")


def lesson_pipeline():
    """Gray -> blur -> threshold -> contour, the four steps everything else sits on."""
    src = cv2.imread(SAMPLES + "ex3.jpg")
    src = cv2.resize(src, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = src.copy()
    keep = [c for c in cnts if cv2.contourArea(c) > 800]
    cv2.drawContours(out, keep, -1, (111, 191, 115), 3)
    cv2.putText(out, "objects: %d" % len(keep), (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (111, 191, 115), 2, cv2.LINE_AA)
    strip([gray, blur, th, out],
          ["1  grayscale", "2  blur", "3  threshold", "4  contours"],
          "ip_pipeline.png",
          )
    return len(keep)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("red discs   :", lesson_colour())
    print("coins       :", lesson_coins_blur())
    print("lines kept  :", lesson_lines())
    print("rects       :", lesson_shape_angle())
    lesson_draw()
    print("objects     :", lesson_pipeline())
