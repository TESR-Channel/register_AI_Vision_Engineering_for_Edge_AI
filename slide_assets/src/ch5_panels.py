"""Figures for chapter 5, Build Your AI Detector.

The chapter ran nineteen content pages, almost all of them walls of cards,
and the sixteen-line webcam loop appeared verbatim on two of them. These
figures carry the ideas that were being described in prose.

Everything with a number on it is measured at render time by running the
model, and printed to stdout when this script runs. Nothing is typed in by
hand, and no frame-rate figure appears anywhere - that is the chapter's own
stated rule and it still holds.

Output: /home/claude/deck/img/ch5_*.png, for a 1920x1080 slide at 2x.
"""
import os
import sys
from collections import Counter

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "/home/claude/deck")
from ch0_panels import font, wrap, OUT, BG, PANEL, EDGE, GOLD, MUTED, WHITE, S, W, H

AI = "/home/claude/deck/ai_samples/"
os.environ.setdefault("YOLO_VERBOSE", "False")

# The same weights the workshop commands on the slides use. If this ever
# diverges from the deck, a learner running the slide's command gets different
# numbers from the figure beside it.
DET_WEIGHTS = "yolo11n.pt"
CLS_WEIGHTS = "yolo11n-cls.pt"

GREEN = (111, 191, 115)
RED = (214, 92, 88)
BLUE = (91, 141, 239)
DIM = (108, 116, 128)

# box colours, BGR for cv2
C_PERSON = (115, 191, 111)
C_BUS = (88, 92, 214)
C_OTHER = (88, 92, 214)
C_WRONG = (88, 92, 214)


def canvas():
    img = Image.new("RGB", (W * S, H * S), BG)
    return img, ImageDraw.Draw(img)


def panel(d, x, y, w, h, ec=EDGE, fc=PANEL):
    d.rounded_rectangle([x * S, y * S, (x + w) * S, (y + h) * S],
                        radius=12 * S, fill=fc, outline=ec, width=2)


def paste_fit(img, d, frame, box, outline=EDGE):
    """Drop a BGR frame into a logical box, letterboxed, with a hairline."""
    x, y, bw, bh = box
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ih, iw = rgb.shape[:2]
    sc = min(bw / iw, bh / ih)
    tw, th = max(1, int(iw * sc)), max(1, int(ih * sc))
    tile = Image.fromarray(cv2.resize(rgb, (tw * S, th * S),
                                      interpolation=cv2.INTER_AREA))
    px = (x + (bw - tw) // 2) * S
    py = (y + (bh - th) // 2) * S
    img.paste(tile, (px, py))
    d.rectangle([px - 2, py - 2, px + tw * S + 1, py + th * S + 1],
                outline=outline, width=2)
    return px, py, tw * S, th * S


def bus_square(raw=None):
    """The bus photograph cropped square, so a row of panels is not mostly gap."""
    raw = cv2.imread(AI + "bus.jpg") if raw is None else raw
    return raw[200:1010, :].copy()


def draw_boxes(img, r, colour=None, thick=4, label=True, label_px=600):
    """Boxes with labels sized for the width this panel will be shown at."""
    out = img.copy()
    for b, c, cf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
        x1, y1, x2, y2 = [int(v) for v in b]
        nm = r.names[int(c)]
        col = colour or (C_PERSON if nm == "person" else C_BUS)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, thick)
        if label:
            tag = "%s %.2f" % (nm, float(cf))
            fs = max(0.6, out.shape[1] / label_px)
            tw = int(17 * fs * len(tag))
            cv2.rectangle(out, (x1, max(y1 - int(32 * fs), 0)),
                          (x1 + tw, y1), col, -1)
            cv2.putText(out, tag, (x1 + int(6 * fs), max(y1 - int(10 * fs), 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs * 0.62, (18, 20, 24),
                        max(1, int(fs * 1.6)))
    return out


def names_of(r):
    return [r.names[int(c)] for c in r.boxes.cls]


def diff_verdict(base, cnt):
    """Describe a result against the baseline by class, not just by count.

    A count that happens to match can still hide a swap - the occlusion case
    loses the bus and invents a fourth person while staying at four.
    """
    lost = [(k, base[k] - cnt.get(k, 0)) for k in base
            if base[k] > cnt.get(k, 0)]
    gained = [(k, cnt[k] - base.get(k, 0)) for k in cnt
              if cnt[k] > base.get(k, 0)]
    n = sum(cnt.values())
    if not lost and not gained:
        return "%d found - unchanged" % n, GREEN
    if not n:
        return "nothing found at all", RED
    bits = []
    if lost:
        bits.append("lost " + ", ".join("%d × %s" % (v, k)
                                        for k, v in lost))
    if gained:
        bits.append("gained " + ", ".join("%d × %s" % (v, k)
                                          for k, v in gained))
    return "%d found - %s" % (n, "; ".join(bits)), RED


def footnote(d, text, fnt, pad=14, start=None):
    ny = H - 60 if start is None else start
    for ln in wrap(d, text, fnt, (W - 2 * pad) * S):
        d.text((pad * S, ny * S), ln, font=fnt, fill=MUTED)
        ny += 30


# --------------------------------------------------------------------------
# ch5_list - a classifier always answers; a detector may answer nothing
# --------------------------------------------------------------------------
def fig_list(det, cls):
    img, d = canvas()
    f_h = font("Kanit", 27, "bold")
    f_r = font("Kanit", 25, "bold")
    f_s = font("Kanit", 22)
    f_n = font("Kanit", 23)

    raw = cv2.imread(AI + "bus.jpg")
    bus = raw[200:1010, :].copy()            # the bus, squared up
    zid = cv2.imread(AI + "zidane.jpg")
    # a piece of plain wall from the same photograph - nothing in it
    wall = raw[0:210, 60:700].copy()

    scenes = [("a busy frame", bus), ("one object", zid), ("an empty wall", wall)]

    PAD, gap = 14, 20
    lab_w = 232
    cw = (W - 2 * PAD - lab_w - 3 * gap) // 3
    col_x = [PAD + lab_w + gap + i * (cw + gap) for i in range(3)]
    ih = 204

    for i, (cap, _) in enumerate(scenes):
        d.text((col_x[i] * S, 10 * S), cap, font=f_s, fill=MUTED)

    # ---- row 1: the detector ----
    ry = 46
    d.text((PAD * S, (ry + 66) * S), "Detector", font=f_h, fill=GOLD)
    d.text((PAD * S, (ry + 104) * S), "a list, one entry", font=f_s, fill=MUTED)
    d.text((PAD * S, (ry + 132) * S), "per object", font=f_s, fill=MUTED)
    for i, (_, frame) in enumerate(scenes):
        r = det(frame, conf=0.5, verbose=False)[0]
        n = len(r.boxes)
        print("  detector  %-14s %d  %s" % (scenes[i][0], n,
                                            dict(Counter(names_of(r)))))
        panel(d, col_x[i], ry, cw, ih, fc=(20, 24, 29))
        paste_fit(img, d, draw_boxes(frame, r), (col_x[i], ry, cw, ih))
        txt = "%d objects" % n if n else "[]   nothing"
        d.text((col_x[i] * S, (ry + ih + 10) * S), txt, font=f_r,
               fill=GREEN if n else GOLD)

    # ---- row 2: the classifier ----
    ry = 310
    d.text((PAD * S, (ry + 66) * S), "Classifier", font=f_h, fill=GOLD)
    d.text((PAD * S, (ry + 104) * S), "one word for the", font=f_s, fill=MUTED)
    d.text((PAD * S, (ry + 132) * S), "whole frame", font=f_s, fill=MUTED)
    for i, (_, frame) in enumerate(scenes):
        rc = cls(frame, verbose=False)[0]
        top = int(rc.probs.top1)
        word = rc.names[top].replace("_", " ")
        conf = float(rc.probs.top1conf)
        print("  classifier %-14s %s %.2f" % (scenes[i][0], word, conf))
        panel(d, col_x[i], ry, cw, ih, fc=(20, 24, 29))
        paste_fit(img, d, frame, (col_x[i], ry, cw, ih))
        last = i == len(scenes) - 1
        d.text((col_x[i] * S, (ry + ih + 10) * S), "%s  %.2f" % (word, conf),
               font=f_r, fill=RED if last else WHITE)

    footnote(d, "Measured on these photographs when the slide was built. The "
                "classifier has no way to say nothing is here - it picks its "
                "best word for a blank wall and sounds sure about it. That is "
                "the reason to detect rather than classify when the answer "
                "might be none.", f_n)
    img.save(OUT + "ch5_list.png")
    print("ch5_list.png")


# --------------------------------------------------------------------------
# ch5_box - what one detection actually is
# --------------------------------------------------------------------------
def fig_box(det):
    img, d = canvas()
    f_h = font("Kanit", 28, "bold")
    f_c = font("DejaVu Sans Mono", 23)
    f_s = font("Kanit", 23)
    f_t = font("Kanit", 21, "bold")
    f_n = font("Kanit", 23)

    src = cv2.imread(AI + "bus.jpg")
    r = det(src, conf=0.5, verbose=False)[0]
    # the strongest detection in the frame
    k = int(np.argmax(r.boxes.conf.cpu().numpy()))
    x1, y1, x2, y2 = [int(v) for v in r.boxes.xyxy[k]]
    cid = int(r.boxes.cls[k])
    name = r.names[cid]
    conf = float(r.boxes.conf[k])
    print("  strongest box: %s idx=%d conf=%.2f xyxy=%d,%d,%d,%d"
          % (name, cid, conf, x1, y1, x2, y2))

    shot = src.copy()
    cv2.rectangle(shot, (x1, y1), (x2, y2), (65, 164, 217), 6)
    for (cx, cy) in [(x1, y1), (x2, y2)]:
        cv2.circle(shot, (cx, cy), 16, (65, 164, 217), -1)
        cv2.circle(shot, (cx, cy), 16, (20, 24, 30), 3)

    PAD = 14
    left_w = 430
    paste_fit(img, d, shot, (PAD, 40, left_w, 500))
    d.text((PAD * S, 8 * S), "one frame, one box", font=f_s, fill=MUTED)

    # the corner callouts, placed next to the picture
    x0 = PAD + left_w + 22
    d.text((x0 * S, 12 * S), "What the model handed back for that box",
           font=f_h, fill=WHITE)

    rows = [
        ("b.xyxy[0]", "[%d, %d, %d, %d]" % (x1, y1, x2, y2),
         "top-left and bottom-right corner, in pixels of the frame you "
         "passed in"),
        ("b.cls[0]", str(cid),
         "a class number, not a word - the model only ever returns the index"),
        ("r.names[%d]" % cid, "'%s'" % name,
         "the names list turns that index into the word your operator reads"),
        ("b.conf[0]", "%.2f" % conf,
         "how strongly this model votes for this box - not how often it is "
         "right"),
    ]
    y = 68
    rh = 108
    for code, val, note in rows:
        panel(d, x0, y, W - PAD - x0, rh - 12)
        d.text(((x0 + 22) * S, (y + 16) * S), code, font=f_c, fill=GOLD)
        d.text(((x0 + 240) * S, (y + 16) * S), val, font=f_c, fill=WHITE)
        yy = y + 52
        for ln in wrap(d, note, f_s, (W - PAD - x0 - 44) * S):
            d.text(((x0 + 22) * S, yy * S), ln, font=f_s, fill=MUTED)
            yy += 28
        y += rh

    d.text((x0 * S, 506 * S), "resize the frame and these numbers change",
           font=f_t, fill=GOLD)
    footnote(d, "Real output, read off this photograph when the slide was "
                "built. Counting, zone alerts, logging and tracking are all "
                "just code that reads this list once per frame.", f_n)
    img.save(OUT + "ch5_box.png")
    print("ch5_box.png")


# --------------------------------------------------------------------------
# ch5_dials - what the confidence dial actually does
# --------------------------------------------------------------------------
def fig_dials(det):
    img, d = canvas()
    f_h = font("Kanit", 30, "bold")
    f_r = font("Kanit", 26, "bold")
    f_s = font("Kanit", 23)
    f_n = font("Kanit", 23)

    # the full frame here, not the square crop - the junk the model is holding
    # back sits in the parts of the scene the crop throws away
    src = cv2.imread(AI + "bus.jpg")
    confs = [0.05, 0.50, 0.90]
    results = []
    for c in confs:
        r = det(src, conf=c, verbose=False)[0]
        results.append((c, r, len(r.boxes), Counter(names_of(r))))
        print("  conf=%.2f -> %d  %s" % (c, len(r.boxes),
                                         dict(Counter(names_of(r)))))
    base_cnt, base_n = results[1][3], results[1][2]

    # no headline here - the slide's own title and subtitle carry it
    top, cw, chh, gap = 14, 330, 440, 50
    x0 = (W - 3 * cw - 2 * gap) // 2
    for i, (c, r, n, cnt) in enumerate(results):
        x = x0 + i * (cw + gap)
        d.text((x * S, (top - 6) * S), "conf = %.2f" % c, font=f_r, fill=GOLD)
        panel(d, x, top + 32, cw, chh, fc=(20, 24, 29))
        paste_fit(img, d, draw_boxes(src, r, label_px=460),
                  (x, top + 32, cw, chh))
        col = GREEN if n == base_n else RED
        d.text((x * S, (top + chh + 44) * S), "%d found" % n, font=f_r,
               fill=col)
        # one short line, so the panel note never runs into the footnote
        if n > base_n:
            note = "%d more than at conf 0.50" % (n - base_n)
        elif n < base_n:
            note = "%d real objects lost" % (base_n - n)
        else:
            note = "the setting the rest of this deck uses"
        d.text((x * S, (top + chh + 80) * S), note, font=f_s, fill=MUTED)

    footnote(d, "Measured on this photograph when the slide was built. Wind "
                "the dial down and the model shows you everything it was "
                "unsure about - look at what those extra boxes are sitting "
                "on. There is no correct default.", f_n, start=566)
    img.save(OUT + "ch5_dials.png")
    print("ch5_dials.png")


# --------------------------------------------------------------------------
# ch5_source - one argument, any source
# --------------------------------------------------------------------------
SOURCES = [
    ("a still image", "source=part.jpg", "one frame, no timing pressure"),
    ("a whole folder", "source=images/", "every file in it, in one run"),
    ("a video file", "source=line.mp4", "motion, and the same frames twice"),
    ("the webcam", "source=0", "live, and now latency is yours"),
    ("an IP camera", "source=rtsp://...", "the same code, a camera on a pole"),
]


def fig_source():
    img, d = canvas()
    f_t = font("Kanit", 30, "bold")
    f_c = font("DejaVu Sans Mono", 26)
    f_cg = font("DejaVu Sans Mono", 26)
    f_h = font("Kanit", 24, "bold")
    f_m = font("DejaVu Sans Mono", 21)
    f_s = font("Kanit", 21)
    f_n = font("Kanit", 23)

    PAD = 14

    # the command, with source= picked out
    cy = 18
    panel(d, PAD, cy, W - 2 * PAD, 118, ec=GOLD, fc=(26, 21, 18))
    parts = [("yolo detect predict  model=yolo11n.pt  ", WHITE),
             ("source=part.jpg", GOLD),
             ("  conf=0.5  save=True", WHITE)]
    tx = PAD + 30
    for txt, col in parts:
        d.text((tx * S, (cy + 38) * S), txt, font=f_c, fill=col)
        tx += d.textlength(txt, font=f_c) / S
    # underline the part that changes
    ux0 = PAD + 30 + d.textlength(parts[0][0], font=f_c) / S
    ux1 = ux0 + d.textlength(parts[1][0], font=f_c) / S
    d.line([ux0 * S, (cy + 80) * S, ux1 * S, (cy + 80) * S], fill=GOLD, width=5)
    d.text((ux0 * S, (cy + 88) * S), "this is the only word that changes",
           font=f_s, fill=GOLD)

    gap = 14
    cw = (W - 2 * PAD - 4 * gap) // 5
    top, chh = 196, 252
    for i, (title, code, note) in enumerate(SOURCES):
        x = PAD + i * (cw + gap)
        panel(d, x, top, cw, chh)
        d.ellipse([(x + 24) * S, (top + 24) * S, (x + 60) * S, (top + 60) * S],
                  outline=GOLD, width=3)
        d.text(((x + 36) * S, (top + 29) * S), str(i + 1), font=f_h, fill=GOLD)
        yy = top + 80
        for ln in wrap(d, title, f_h, (cw - 48) * S):
            d.text(((x + 24) * S, yy * S), ln, font=f_h, fill=WHITE)
            yy += 32
        yy += 10
        for ln in wrap(d, code, f_m, (cw - 48) * S):
            d.text(((x + 24) * S, yy * S), ln, font=f_m, fill=GOLD)
            yy += 28
        yy += 10
        for ln in wrap(d, note, f_s, (cw - 48) * S):
            d.text(((x + 24) * S, yy * S), ln, font=f_s, fill=MUTED)
            yy += 28

    footnote(d, "Everything you learn about the other arguments carries "
                "across all five. Work through them in order - each one adds "
                "exactly one new thing, and the last two are where a demo "
                "starts behaving like an installation.", f_n)
    img.save(OUT + "ch5_source.png")
    print("ch5_source.png")


# --------------------------------------------------------------------------
# ch5_size - what shrinking the input actually costs
# --------------------------------------------------------------------------
def fig_size(det):
    img, d = canvas()
    f_h = font("Kanit", 30, "bold")
    f_r = font("Kanit", 26, "bold")
    f_s = font("Kanit", 23)
    f_n = font("Kanit", 23)

    src = bus_square()
    sizes = [640, 320, 96]
    out = []
    for sz in sizes:
        r = det(src, conf=0.5, imgsz=sz, verbose=False)[0]
        cnt = Counter(names_of(r))
        out.append((sz, r, len(r.boxes), cnt))
        print("  imgsz=%3d -> %d  %s" % (sz, len(r.boxes), dict(cnt)))

    top, cell, gap = 26, 400, 48
    x0 = (W - 3 * cell - 2 * gap) // 2
    base_cnt = out[0][3]
    for i, (sz, r, n, cnt) in enumerate(out):
        x = x0 + i * (cell + gap)
        d.text((x * S, (top - 6) * S), "imgsz = %d" % sz, font=f_r, fill=GOLD)
        panel(d, x, top + 32, cell, cell, fc=(20, 24, 29))
        paste_fit(img, d, draw_boxes(src, r), (x, top + 32, cell, cell))
        txt, col = diff_verdict(base_cnt, cnt)
        yy = top + cell + 44
        for ln in wrap(d, txt, f_s, cell * S):
            d.text((x * S, yy * S), ln, font=f_s, fill=col)
            yy += 30

    footnote(d, "Measured on this photograph when the slide was built. Input "
                "size is the strongest lever you have on speed, and this is "
                "what it costs - here you can go a long way down before "
                "anything is lost, then it collapses. Where your own scene "
                "breaks is a number only you can find.", f_n)
    img.save(OUT + "ch5_size.png")
    print("ch5_size.png")


# --------------------------------------------------------------------------
# ch5_break - four ways to break it on purpose
# --------------------------------------------------------------------------
def fig_break(det):
    img, d = canvas()
    f_h = font("Kanit", 30, "bold")
    f_t = font("Kanit", 25, "bold")
    f_r = font("Kanit", 25, "bold")
    f_s = font("Kanit", 22)
    f_n = font("Kanit", 23)

    src = bus_square()
    h, w = src.shape[:2]
    base_cnt = Counter(names_of(det(src, conf=0.5, verbose=False)[0]))
    base = sum(base_cnt.values())
    print("  baseline -> %d  %s" % (base, dict(base_cnt)))

    # 1. half the object behind something
    occl = src.copy()
    cv2.rectangle(occl, (0, int(h * 0.30)), (w, int(h * 0.62)), (22, 24, 28), -1)

    # 2. the same scene, much further away
    small = np.full_like(src, 118)
    sc = 0.12
    tiny = cv2.resize(src, (int(w * sc), int(h * sc)),
                      interpolation=cv2.INTER_AREA)
    oy, ox = (h - tiny.shape[0]) // 2, (w - tiny.shape[1]) // 2
    small[oy:oy + tiny.shape[0], ox:ox + tiny.shape[1]] = tiny

    # 3. the camera mounted on its side
    rot = cv2.rotate(src, cv2.ROTATE_90_CLOCKWISE)

    # 4. a moving line, or a camera that cannot hold still
    k = 45
    kern = np.zeros((k, k), np.float32)
    kern[k // 2, :] = 1.0 / k
    blur = cv2.filter2D(src, -1, kern)

    cases = [
        ("Cover half of it", occl, 0.5,
         "a bar across the middle of every object"),
        ("Move it far away", small, 0.5,
         "the same scene, much smaller in frame"),
        ("Mount the camera sideways", rot, 0.5,
         "the same photo, turned a quarter turn"),
        ("Let it move", blur, 0.5,
         "motion blur, as on a running line"),
    ]

    top, cell, gap = 24, 360, 30
    x0 = (W - 4 * cell - 3 * gap) // 2
    for i, (title, frame, c, note) in enumerate(cases):
        x = x0 + i * (cell + gap)
        d.text((x * S, (top - 6) * S), title, font=f_t, fill=GOLD)
        r = det(frame, conf=c, verbose=False)[0]
        cnt = Counter(names_of(r))
        n = len(r.boxes)
        print("  %-26s -> %d  %s" % (title, n, dict(cnt)))
        panel(d, x, top + 32, cell, cell, fc=(20, 24, 29))
        paste_fit(img, d, draw_boxes(frame, r, thick=5),
                  (x, top + 32, cell, cell))
        if c < 0.5:
            col, txt = GOLD, "%d found, against %d at 0.50" % (n, base)
        else:
            txt, col = diff_verdict(base_cnt, cnt)
        yy = top + cell + 44
        for ln in wrap(d, txt, f_r, cell * S):
            d.text((x * S, yy * S), ln, font=f_r, fill=col)
            yy += 30
        yy += 4
        for ln in wrap(d, note, f_s, cell * S):
            d.text((x * S, yy * S), ln, font=f_s, fill=MUTED)
            yy += 26

    footnote(d, "Measured when the slide was built, against %d objects found "
                "on the untouched photograph. Write down the two failures "
                "that surprise you most - they are the first entries in your "
                "own dataset." % base, f_n)
    img.save(OUT + "ch5_break.png")
    print("ch5_break.png")


if __name__ == "__main__":
    from ultralytics import YOLO
    det = YOLO(DET_WEIGHTS)
    cls = YOLO(CLS_WEIGHTS)
    print("ch5_list:")
    fig_list(det, cls)
    print("ch5_box:")
    fig_box(det)
    print("ch5_dials:")
    fig_dials(det)
    print("ch5_source:")
    fig_source()
    print("ch5_size:")
    fig_size(det)
    print("ch5_break:")
    fig_break(det)
