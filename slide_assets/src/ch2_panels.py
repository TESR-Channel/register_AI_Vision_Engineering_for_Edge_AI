"""Figures for chapter 2, the AI Vision Workflow.

The chapter used to be eight pages of prose with nothing to look at. These
four figures carry it instead, and every number on them is produced by
actually running OpenCV on the course's own sample images - the same files
that ship in the Image Processing Lab. Nothing is drawn to look like a result.

The one exception is the pipeline diagram, which is a diagram and says so.

Output: /home/claude/deck/img/ch2_*.png, sized for a 1920x1080 slide at 2x.
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm

SAMPLES = "/home/claude/image_processing_lab/samples/"
OUT = "/home/claude/deck/img/"

BG = (15, 18, 22)
PANEL = (27, 32, 39)
EDGE = (58, 67, 79)
GOLD = (217, 164, 65)
GREEN = (111, 191, 115)
RED = (214, 92, 88)
MUTED = (154, 163, 174)
WHITE = (240, 240, 242)

S = 2
W, H = 1676, 620


def font(name, px, weight="normal"):
    p = fm.findfont(fm.FontProperties(family=name, weight=weight),
                    fallback_to_default=True)
    return ImageFont.truetype(p, int(px * S))


def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else \
           cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def canvas():
    img = Image.new("RGB", (W * S, H * S), BG)
    return img, ImageDraw.Draw(img)


def wrap(d, text, fnt, maxw):
    """Greedy wrap to a pixel width, in device px."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=fnt) <= maxw or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def paste_fit(img, d, frame, box, outline=EDGE):
    """Fit a BGR frame inside (x, y, w, h) slide px, centred. Returns bbox."""
    x, y, bw, bh = box
    rgb = bgr2rgb(frame)
    ih, iw = rgb.shape[:2]
    sc = min(bw / iw, bh / ih)
    tw, th = int(iw * sc), int(ih * sc)
    tile = Image.fromarray(cv2.resize(rgb, (tw * S, th * S),
                                      interpolation=cv2.INTER_AREA))
    px = (x + (bw - tw) // 2) * S
    py = (y + (bh - th) // 2) * S
    img.paste(tile, (px, py))
    d.rectangle([px - 2, py - 2, px + tw * S + 1, py + th * S + 1],
                outline=outline, width=2)
    return px, py, tw * S, th * S


# --------------------------------------------------------------------------
# helpers that produce the real results
# --------------------------------------------------------------------------
def red_discs(hue_hi):
    """Find red discs on ex3.png. hue_hi widens the mask into orange."""
    img = cv2.imread(SAMPLES + "ex3.png")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 120, 70), (hue_hi, 255, 255)) | \
        cv2.inRange(hsv, (172, 120, 70), (179, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return img, [c for c in cs if cv2.contourArea(c) > 400]


def boxed(img, cs, colour):
    out = img.copy()
    for c in cs:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(out, (x, y), (x + w, y + h), colour, 3)
    return out


def tally_card(rows, foot_lines, cw, ch):
    """Draw a card at exactly the cell size so nothing is scaled or clipped."""
    im = Image.new("RGB", (cw * S, ch * S), PANEL)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, cw * S - 1, ch * S - 1], outline=EDGE, width=2)
    f_k = font("Kanit", 29)
    f_v = font("Kanit", 40, "bold")
    f_f = font("Kanit", 24)
    M = 30

    y = 34
    for k, v, col in rows:
        d.text((M * S, y * S), k, font=f_k, fill=MUTED)
        d.text(((cw - M) * S - d.textlength(v, font=f_v), (y - 8) * S),
               v, font=f_v, fill=col)
        y += 62
        d.line([M * S, (y - 16) * S, (cw - M) * S, (y - 16) * S],
               fill=EDGE, width=2)
    y += 18
    for ln in foot_lines:
        for seg in wrap(d, ln, f_f, (cw - 2 * M) * S):
            d.text((M * S, y * S), seg, font=f_f, fill=MUTED)
            y += 32
        y += 10
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------
# A - the eight stages, drawn as one loop
# --------------------------------------------------------------------------
STAGES = [
    ("1", "Define",   "problem statement"),
    ("2", "Collect",  "raw images"),
    ("3", "Annotate", "labelled dataset"),
    ("4", "Train",    "model weights"),
    ("5", "Evaluate", "metrics report"),
    ("6", "Optimise", "ONNX / engine"),
    ("7", "Deploy",   "running system"),
    ("8", "Monitor",  "field data"),
]


def fig_loop():
    img, d = canvas()
    f_n = font("Kanit", 30, "bold")
    f_s = font("Kanit", 33, "bold")
    f_a = font("Kanit", 23)
    f_c = font("Kanit", 26)

    PAD, gap, arrow = 14, 12, 22
    n = len(STAGES)
    bw = (W - 2 * PAD - (n - 1) * (gap + arrow)) // n
    bh, top = 296, 86

    centres = []
    x = PAD
    for num, name, art in STAGES:
        d.rounded_rectangle([x * S, top * S, (x + bw) * S, (top + bh) * S],
                            radius=10 * S, fill=PANEL, outline=EDGE, width=2)
        d.text(((x + 18) * S, (top + 26) * S), num, font=f_n, fill=GOLD)
        d.text(((x + 18) * S, (top + 86) * S), name, font=f_s, fill=WHITE)
        yy = top + 152
        for seg in wrap(d, art, f_a, (bw - 36) * S):
            d.text(((x + 18) * S, yy * S), seg, font=f_a, fill=MUTED)
            yy += 30
        centres.append(x + bw // 2)
        if num != "8":
            ax = (x + bw + gap // 2 + 2) * S
            ay = (top + bh // 2) * S
            d.polygon([(ax, ay - 13 * S), (ax + 22 * S, ay),
                       (ax, ay + 13 * S)], fill=GOLD)
        x += bw + gap + arrow

    # the return leg: Monitor feeds Collect
    y0 = (top + bh + 14) * S
    y1 = (top + bh + 74) * S
    x8, x2 = centres[7] * S, centres[1] * S
    d.line([x8, y0, x8, y1], fill=GOLD, width=4)
    d.line([x8, y1, x2, y1], fill=GOLD, width=4)
    d.line([x2, y1, x2, y0 + 6], fill=GOLD, width=4)
    d.polygon([(x2 - 13 * S, y0 + 22 * S), (x2 + 13 * S, y0 + 22 * S),
               (x2, y0)], fill=GOLD)

    cap = ("Monitor feeds Collect. The frames the model was unsure about "
           "are your next dataset.")
    d.text((PAD * S, (top + bh + 92) * S), cap, font=f_c, fill=GOLD)
    note = ("A diagram, not a measurement. Skipping a stage does not remove "
            "its cost - it moves it to a later one.")
    d.text((PAD * S, (top + bh + 148) * S), note, font=f_a, fill=MUTED)

    img.save(OUT + "ch2_loop.png")
    print("ch2_loop.png")


# --------------------------------------------------------------------------
# B - the four task types, each a real result
# --------------------------------------------------------------------------
def fig_tasks():
    # the same frame answers the first three questions, so the difference
    # the learner sees is the question, not the picture
    disc, cs = red_discs(4)

    # 1. classification - one label for the whole frame
    cls = disc.copy()
    bh_ = max(34, disc.shape[0] // 11)
    cv2.rectangle(cls, (0, 0), (cls.shape[1], bh_), (24, 24, 24), -1)
    cv2.putText(cls, "whole frame -> 1 label", (12, int(bh_ * 0.74)),
                cv2.FONT_HERSHEY_SIMPLEX, bh_ / 46, (240, 240, 242), 2)

    # 2. detection - the six red discs, boxed
    det = boxed(disc, cs, (88, 92, 214))

    # 3. segmentation - the pixels themselves, not a box
    hsv = cv2.cvtColor(disc, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 120, 70), (4, 255, 255)) | \
        cv2.inRange(hsv, (172, 120, 70), (179, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    grey = cv2.cvtColor(cv2.cvtColor(disc, cv2.COLOR_BGR2GRAY),
                        cv2.COLOR_GRAY2BGR)
    grey = cv2.convertScaleAbs(grey, alpha=0.55, beta=12)
    lime = np.zeros_like(disc)
    lime[:] = (60, 230, 90)
    seg = np.where(m[:, :, None].astype(bool),
                   cv2.addWeighted(disc, 0.25, lime, 0.75, 0), grey)
    sc, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(seg, [c for c in sc if cv2.contourArea(c) > 400],
                     -1, (40, 255, 70), 2)

    # 4. reading a value - decoded for real
    code = cv2.imread(SAMPLES + "codes_barcode.jpg")
    ok, info, typ, _ = cv2.barcode.BarcodeDetector().detectAndDecodeWithType(
        cv2.cvtColor(code, cv2.COLOR_BGR2GRAY))
    value = list(info)[0] if ok and any(info) else ""
    kind = list(typ)[0] if ok and any(typ) else ""
    cod = code.copy()
    cv2.rectangle(cod, (0, 0), (cod.shape[1], 86), (24, 24, 24), -1)
    cv2.putText(cod, value, (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.7,
                (217, 164, 65), 4)

    frames = [cls, det, seg, cod]
    labels = ["Classification", "Detection", "Segmentation", "Read a value"]
    subs = ["which category is this?", "what is present, and where?",
            "which exact pixels?",
            kind.replace("_", "-") + ", decoded"]

    img, d = canvas()
    f_l = font("Kanit", 29, "bold")
    f_s = font("Kanit", 25)
    f_n = font("Kanit", 25)

    PAD, gap = 14, 34
    cw = (W - 2 * PAD - 3 * gap) // 4
    top, chh = 52, 392
    x = PAD
    for fr, lab, sub in zip(frames, labels, subs):
        d.text((x * S, (top - 34) * S), lab, font=f_l, fill=GOLD)
        paste_fit(img, d, fr, (x, top, cw, chh))
        d.text((x * S, (top + chh + 10) * S), sub, font=f_s, fill=MUTED)
        x += cw + gap

    d.text((PAD * S, (H - 44) * S),
           "Four different questions you can ask of one camera. The decision "
           "you need picks the task - and the task picks the dataset.",
           font=f_n, fill=MUTED)
    img.save(OUT + "ch2_tasks.png")
    print("ch2_tasks.png")


# --------------------------------------------------------------------------
# C - a real false-alarm run, which is what precision and recall mean
# --------------------------------------------------------------------------
def fig_errors():
    loose_img, loose = red_discs(15)
    tight_img, tight = red_discs(4)
    n_loose, n_tight = len(loose), len(tight)
    false_alarms = n_loose - n_tight

    a = boxed(loose_img, loose, (88, 92, 214))
    b = boxed(tight_img, tight, (115, 191, 111))

    img, d = canvas()
    f_l = font("Kanit", 29, "bold")
    f_s = font("Kanit", 25)
    f_n = font("Kanit", 25)

    PAD, gap = 14, 34
    cw = (W - 2 * PAD - 2 * gap) // 3
    top, chh = 46, 436

    card = tally_card(
        [("Discs flagged", str(n_loose), RED),
         ("Actually red", str(n_tight), GREEN),
         ("False alarms", str(false_alarms), RED)],
        ["Precision: of what it flagged, how much was real.",
         "Recall: of what was really there, how much it caught."],
        cw, chh - 16)
    cells = [
        (a, "Hue 0-15: too wide", "%d found - orange counted as red" % n_loose),
        (b, "Hue 0-4 and 172-179", "%d found - these are the red ones" % n_tight),
        (card, "What that costs you", "same image, same code, one number changed"),
    ]
    x = PAD
    for fr, lab, sub in cells:
        d.text((x * S, (top - 40) * S), lab, font=f_l, fill=GOLD)
        paste_fit(img, d, fr, (x, top, cw, chh))
        d.text((x * S, (top + chh + 10) * S), sub, font=f_s, fill=MUTED)
        x += cw + gap

    d.text((PAD * S, (H - 44) * S),
           "Measured on the course sample ex3.png. Which error hurts more is "
           "a business decision, not a technical one.",
           font=f_n, fill=MUTED)
    img.save(OUT + "ch2_errors.png")
    print("ch2_errors.png  loose=%d tight=%d false=%d"
          % (n_loose, n_tight, false_alarms))


# --------------------------------------------------------------------------
# D - export chain on top, what changes in the field underneath
# --------------------------------------------------------------------------
CHAIN = [("PyTorch", ".pt from training"),
         ("ONNX", ".onnx, portable"),
         ("Runtime engine", "TensorRT / TFLite / OpenVINO / Hailo"),
         ("Edge device", "running on the line")]


def fig_field():
    img, d = canvas()
    f_c = font("Kanit", 29, "bold")
    f_cs = font("Kanit", 22)
    f_l = font("Kanit", 28, "bold")
    f_s = font("Kanit", 24)
    f_n = font("Kanit", 25)

    # top band: one model, several forms
    PAD, gap, arrow = 14, 12, 26
    n = len(CHAIN)
    bw = (W - 2 * PAD - (n - 1) * (gap + arrow)) // n
    bh, top = 118, 30
    x = PAD
    for i, (name, sub) in enumerate(CHAIN):
        d.rounded_rectangle([x * S, top * S, (x + bw) * S, (top + bh) * S],
                            radius=10 * S, fill=PANEL, outline=EDGE, width=2)
        d.text(((x + 20) * S, (top + 20) * S), name, font=f_c,
               fill=GOLD if i == n - 1 else WHITE)
        yy = top + 60
        for seg in wrap(d, sub, f_cs, (bw - 40) * S):
            d.text(((x + 20) * S, yy * S), seg, font=f_cs, fill=MUTED)
            yy += 28
        if i < n - 1:
            ax = (x + bw + gap // 2 + 2) * S
            ay = (top + bh // 2) * S
            d.polygon([(ax, ay - 13 * S), (ax + 24 * S, ay),
                       (ax, ay + 13 * S)], fill=GOLD)
        x += bw + gap + arrow
    d.text((PAD * S, (top + bh + 12) * S),
           "Re-test after every conversion. Each step can change the output.",
           font=f_s, fill=GOLD)

    # bottom band: the same scene, changed three ways
    base = cv2.imread(SAMPLES + "ex3.png")
    h0, w0 = base.shape[:2]

    dark = cv2.convertScaleAbs(base, alpha=0.45, beta=-10)      # lamp off
    M = cv2.getRotationMatrix2D((w0 / 2, h0 / 2), 11, 1.0)
    M[0, 2] += 38
    M[1, 2] += 22
    moved = cv2.warpAffine(base, M, (w0, h0), borderValue=(18, 18, 18))
    hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + 60) % 180
    recol = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)               # new packaging

    cells = [(dark, "The scene changed", "a lamp off, a shift that leaves the door open"),
             (moved, "The camera moved", "bumped during cleaning, nobody noticed"),
             (recol, "The product changed", "new packaging colour, never seen in training")]

    PAD2, gap2 = 14, 40
    cw = (W - 2 * PAD2 - 2 * gap2) // 3
    ytop, chh = 254, 268
    x = PAD2
    for fr, lab, sub in cells:
        d.text((x * S, (ytop - 34) * S), lab, font=f_l, fill=WHITE)
        bx, by, bwid, bhgt = paste_fit(img, d, base, (x, ytop, cw // 2 - 8, chh))
        paste_fit(img, d, fr, (x + cw // 2 + 8, ytop, cw // 2 - 8, chh),
                  outline=GOLD)
        d.text((x * S, (ytop + chh + 8) * S), sub, font=f_s, fill=MUTED)
        x += cw + gap2

    d.text((PAD2 * S, (H - 40) * S),
           "Left: the sample as trained on. Right: the same file, altered in "
           "code to stand in for a real change. Build logging and alerting in "
           "from day one.",
           font=f_n, fill=MUTED)
    img.save(OUT + "ch2_field.png")
    print("ch2_field.png")


# --------------------------------------------------------------------------
# E - stages 2 and 3: what to collect, and labelling the same thing twice
# --------------------------------------------------------------------------
def fig_data():
    img, d = canvas()
    f_b = font("Kanit", 30, "bold")
    f_l = font("Kanit", 25)
    f_s = font("Kanit", 24)
    f_n = font("Kanit", 25)

    base = cv2.imread(SAMPLES + "ex3.png")
    h0, w0 = base.shape[:2]

    # top band - the variation a dataset has to contain
    dim = cv2.convertScaleAbs(base, alpha=0.42, beta=-8)
    hot = cv2.convertScaleAbs(base, alpha=1.34, beta=34)
    blur = cv2.GaussianBlur(base, (0, 0), 4.5)
    variants = [(base, "as collected"), (dim, "lamp off"),
                (hot, "sun on the bench"), (blur, "moving under the camera")]

    d.text((14 * S, 16 * S), "Stage 2  Collect", font=f_b, fill=GOLD)
    d.text((14 * S, 56 * S),
           "A model cannot recognise a condition that never appeared in "
           "its training data.", font=f_l, fill=MUTED)

    PAD, gap = 14, 30
    cw = (W - 2 * PAD - 3 * gap) // 4
    ytop, chh = 100, 184
    x = PAD
    for fr, lab in variants:
        paste_fit(img, d, fr, (x, ytop, cw, chh))
        d.text((x * S, (ytop + chh + 8) * S), lab, font=f_s, fill=MUTED)
        x += cw + gap

    # bottom band - three labellers, one object
    d.text((14 * S, 338 * S), "Stage 3  Annotate", font=f_b, fill=GOLD)
    d.text((14 * S, 378 * S),
           "Two people labelling the same image differently is a bug in "
           "the dataset.", font=f_l, fill=MUTED)

    _, cs = red_discs(4)
    c = max(cs, key=cv2.contourArea)
    bx, by, bw_, bh_ = cv2.boundingRect(c)
    pad = 46
    x0, y0 = max(bx - pad, 0), max(by - pad, 0)
    x1, y1 = min(bx + bw_ + pad, w0), min(by + bh_ + pad, h0)
    crop = base[y0:y1, x0:x1].copy()
    ox, oy = bx - x0, by - y0

    tight = crop.copy()
    cv2.rectangle(tight, (ox, oy), (ox + bw_, oy + bh_), (115, 191, 111), 3)
    loose = crop.copy()
    cv2.rectangle(loose, (ox - 26, oy - 24), (ox + bw_ + 26, oy + bh_ + 24),
                  (88, 92, 214), 3)
    none_ = crop.copy()

    treats = [(tight, "Tight to the edge", "the rule most teams agree on"),
              (loose, "Loose, drifting", "a tired labeller, late in the batch"),
              (none_, "Not labelled at all", "taught to the model as background")]

    cw2 = (W - 2 * PAD - 2 * gap) // 3
    ytop2, chh2 = 416, 112
    x = PAD
    for fr, lab, sub in treats:
        bxx, byy, bww, bhh = paste_fit(img, d, fr, (x, ytop2, cw2, chh2))
        tx = (bxx + bww) // S + 20
        d.text((tx * S, (ytop2 + 16) * S), lab, font=f_s, fill=WHITE)
        for i, seg in enumerate(wrap(d, sub, f_s, (x + cw2 - tx) * S)):
            d.text((tx * S, (ytop2 + 52 + i * 32) * S), seg,
                   font=f_s, fill=MUTED)
        x += cw2 + gap

    d.text((PAD * S, (H - 40) * S),
           "The same disc from the course sample, boxed three ways. Write the "
           "rule down before anyone starts, or the dataset records the "
           "disagreement.", font=f_n, fill=MUTED)
    img.save(OUT + "ch2_data.png")
    print("ch2_data.png")


if __name__ == "__main__":
    fig_loop()
    fig_tasks()
    fig_errors()
    fig_field()
