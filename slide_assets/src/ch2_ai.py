"""Chapter 2 figures, rebuilt around one subject that actually needs AI.

The first version used the course's coloured-disc photo throughout. Fair
criticism: finding coloured discs is a hue threshold. Nothing on those figures
needed a model, so the chapter never showed why a model is needed.

Every figure here runs a real YOLOv8n over the same street photograph, so the
chapter follows one subject from "what task is this" to "what breaks in the
field". Every label, box, mask and count is model output captured at render
time. Nothing is drawn to look like a result.

Measured once and checked by eye before writing the captions:
  conf 0.50 -> bus 0.87, person 0.87 / 0.85 / 0.83          4 objects
  conf 0.25 -> the same, plus person 0.26 (a real person half
               out of frame on the left) and stop sign 0.26
               (a small sign on the wall - a false alarm)    6 objects
So the high threshold is precise but misses a person; the low one finds
everyone but invents a stop sign. That is the precision/recall trade-off with
real numbers rather than a formula.

Image sources: the cat is scikit-image's `chelsea` (CC0). The street photo is
`bus.jpg` from the Ultralytics assets repository - a widely used demo image,
not a public-domain one. Swap it if the deck needs a cleared photograph.

Output: /home/claude/deck/img/ch2_*.png, for a 1920x1080 slide at 2x.
"""
import os
import sys
import warnings
from collections import Counter

import cv2
import numpy as np
from PIL import Image, ImageDraw

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/claude/deck")
from ch0_panels import font, wrap, OUT, BG, PANEL, EDGE, GOLD, MUTED, WHITE, S, W, H

AI = "/home/claude/deck/ai_samples/"
GREEN = (111, 191, 115)
RED = (214, 92, 88)
BLUE = (65, 164, 217)

# cv2 draws in BGR; these are the on-image colours
C_PERSON = (115, 191, 111)
C_BUS = (88, 92, 214)
C_FALSE = (65, 164, 217)


def rgb(c):
    return (c[2], c[1], c[0])


def canvas():
    img = Image.new("RGB", (W * S, H * S), BG)
    return img, ImageDraw.Draw(img)


def bgr2rgb(a):
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB) if a.ndim == 3 else \
           cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)


def paste_fit(img, d, frame, box, outline=EDGE):
    x, y, bw, bh = box
    r = bgr2rgb(frame)
    ih, iw = r.shape[:2]
    sc = min(bw / iw, bh / ih)
    tw, th = int(iw * sc), int(ih * sc)
    tile = Image.fromarray(cv2.resize(r, (tw * S, th * S),
                                      interpolation=cv2.INTER_AREA))
    px = (x + (bw - tw) // 2) * S
    py = (y + (bh - th) // 2) * S
    img.paste(tile, (px, py))
    d.rectangle([px - 2, py - 2, px + tw * S + 1, py + th * S + 1],
                outline=outline, width=2)
    return px, py, tw * S, th * S


def crop_to_ar(im, ar):
    """Centre-crop to a target width/height ratio, so every cell fills."""
    h, w = im.shape[:2]
    if w / h > ar:
        nw = int(round(h * ar))
        x = (w - nw) // 2
        return im[:, x:x + nw]
    nh = int(round(w / ar))
    y = (h - nh) // 2
    return im[y:y + nh, :]


def draw_boxes(im, res, colour_for, label=True, thick=4):
    out = im.copy()
    for b, c, s in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
        x1, y1, x2, y2 = [int(v) for v in b]
        nm = res.names[int(c)]
        col = colour_for(nm)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, thick)
        if label:
            txt = "%s %.2f" % (nm, float(s))
            cv2.rectangle(out, (x1, max(y1 - 30, 0)),
                          (x1 + 16 + 15 * len(txt), y1), col, -1)
            cv2.putText(out, txt, (x1 + 8, max(y1 - 9, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (18, 20, 24), 2)
    return out


def card(cw, ch, rows, foot, head=None):
    im = Image.new("RGB", (cw * S, ch * S), PANEL)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, cw * S - 1, ch * S - 1], outline=EDGE, width=2)
    f_h = font("Kanit", 27, "bold")
    f_k = font("Kanit", 28)
    f_v = font("Kanit", 40, "bold")
    f_f = font("Kanit", 22)
    M = 28
    y = 26
    if head:
        d.text((M * S, y * S), head, font=f_h, fill=GOLD)
        y += 48
    for name, val, col in rows:
        d.text((M * S, y * S), name, font=f_k, fill=col)
        d.text(((cw - M) * S - d.textlength(str(val), font=f_v), (y - 8) * S),
               str(val), font=f_v, fill=WHITE)
        y += 54
        d.line([M * S, (y - 14) * S, (cw - M) * S, (y - 16) * S],
               fill=EDGE, width=2)
    y += 14
    for ln in foot:
        for seg in wrap(d, ln, f_f, (cw - 2 * M) * S):
            d.text((M * S, y * S), seg, font=f_f, fill=MUTED)
            y += 29
        y += 8
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def colour_for(nm):
    return {"person": C_PERSON, "bus": C_BUS}.get(nm, C_FALSE)


SRC_NOTE = ("Real YOLOv8n output, captured when this slide was built. Cat: "
            "scikit-image `chelsea`. Street photo: the Ultralytics demo image.")


# ==========================================================================
def fig_tasks(det, seg, cls):
    base = cv2.imread(AI + "bus.jpg")
    rd = det(AI + "bus.jpg", conf=0.5, verbose=False)[0]
    names = [rd.names[int(c)] for c in rd.boxes.cls]

    rc = cls(AI + "chelsea.jpg", verbose=False)[0]
    label = rc.names[int(rc.probs.top1)]

    ar = base.shape[1] / base.shape[0]
    cat = crop_to_ar(cv2.imread(AI + "chelsea.jpg"), ar)
    bh = max(40, cat.shape[0] // 11)
    cv2.rectangle(cat, (0, 0), (cat.shape[1], bh), (24, 24, 24), -1)
    cv2.putText(cat, "whole frame ->", (10, int(bh * 0.70)),
                cv2.FONT_HERSHEY_SIMPLEX, bh / 46, (240, 240, 242), 2)
    cv2.putText(cat, label, (10, int(bh * 1.82)),
                cv2.FONT_HERSHEY_SIMPLEX, bh / 42, (65, 164, 217), 2)
    cv2.rectangle(cat, (0, 0), (cat.shape[1], int(bh * 2.1)), (24, 24, 24), -1)
    cv2.putText(cat, "whole frame ->", (10, int(bh * 0.74)),
                cv2.FONT_HERSHEY_SIMPLEX, bh / 46, (240, 240, 242), 2)
    cv2.putText(cat, label, (10, int(bh * 1.76)),
                cv2.FONT_HERSHEY_SIMPLEX, bh / 40, (217, 164, 65), 2)

    detimg = draw_boxes(base, rd, colour_for, label=False, thick=4)
    for b, c in zip(rd.boxes.xyxy, rd.boxes.cls):
        x1, y1 = int(b[0]), int(b[1])
        nm = rd.names[int(c)]
        col = colour_for(nm)
        cv2.rectangle(detimg, (x1, max(y1 - 30, 0)),
                      (x1 + 16 + 16 * len(nm), y1), col, -1)
        cv2.putText(detimg, nm, (x1 + 8, max(y1 - 9, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.66, (18, 20, 24), 2)

    rs = seg(AI + "bus.jpg", conf=0.5, verbose=False)[0]
    grey = cv2.cvtColor(cv2.cvtColor(base, cv2.COLOR_BGR2GRAY),
                        cv2.COLOR_GRAY2BGR)
    segimg = cv2.convertScaleAbs(grey, alpha=0.55, beta=12)
    for m, c in zip(rs.masks.data, rs.boxes.cls):
        nm = rs.names[int(c)]
        mk = cv2.resize(m.cpu().numpy().astype(np.uint8),
                        (base.shape[1], base.shape[0]),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
        tint = np.zeros_like(base)
        tint[:] = colour_for(nm)
        segimg = np.where(mk[:, :, None],
                          cv2.addWeighted(base, 0.3, tint, 0.7, 0), segimg)

    counts = sorted(Counter(names).items(), key=lambda t: -t[1])

    img, d = canvas()
    f_l = font("Kanit", 29, "bold")
    f_s = font("Kanit", 25)
    f_n = font("Kanit", 25)
    PAD, gap = 14, 34
    cw = (W - 2 * PAD - 3 * gap) // 4
    top, chh = 52, 396

    tally = card(cw, chh,
                 [(n, v, rgb(colour_for(n))) for n, v in counts],
                 ["A colour threshold cannot tell a person from a bus.",
                  "This is where a model earns its place."])

    cells = [(cat, "Classification", "one answer for the whole frame"),
             (detimg, "Detection", "what is present, and where"),
             (segimg, "Segmentation", "which pixels, not a box"),
             (tally, "Count it", "a number something can act on")]
    x = PAD
    for fr, lab, sub in cells:
        d.text((x * S, (top - 38) * S), lab, font=f_l, fill=GOLD)
        paste_fit(img, d, fr, (x, top, cw, chh))
        d.text((x * S, (top + chh + 12) * S), sub, font=f_s, fill=MUTED)
        x += cw + gap

    ny = H - 74
    for ln in wrap(d, SRC_NOTE + " The decision you need picks the task.",
                   f_n, (W - 2 * PAD) * S):
        d.text((PAD * S, ny * S), ln, font=f_n, fill=MUTED)
        ny += 34
    img.save(OUT + "ch2_tasks.png")
    print("ch2_tasks.png  cls=%s  counts=%s" % (label, counts))


# ==========================================================================
def fig_data(det):
    base = cv2.imread(AI + "bus.jpg")
    h, w = base.shape[:2]

    dusk = cv2.convertScaleAbs(base, alpha=0.38, beta=-12)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 13, 1.0)
    M[0, 2] += 70
    M[1, 2] += 40
    moved = cv2.warpAffine(base, M, (w, h), borderValue=(18, 18, 18))
    blur = cv2.GaussianBlur(base, (0, 0), 6)

    # the photo is portrait; a wide band through the subjects reads far
    # better at thumbnail size than the whole frame letterboxed
    def band(im):
        return im[int(h * 0.26):int(h * 0.84), :]
    base_b, dusk, moved, blur = band(base), band(dusk), band(moved), band(blur)

    img, d = canvas()
    f_b = font("Kanit", 30, "bold")
    f_l = font("Kanit", 25)
    f_s = font("Kanit", 24)
    f_n = font("Kanit", 25)

    d.text((14 * S, 14 * S), "Stage 2  Collect", font=f_b, fill=GOLD)
    d.text((14 * S, 54 * S),
           "A model cannot recognise a condition that never appeared in its "
           "training data.", font=f_l, fill=MUTED)

    PAD, gap = 14, 30
    cw = (W - 2 * PAD - 3 * gap) // 4
    ytop, chh = 92, 168
    x = PAD
    for fr, lab in [(base_b, "as collected"), (dusk, "dusk"),
                    (moved, "camera moved"), (blur, "motion blur")]:
        paste_fit(img, d, fr, (x, ytop, cw, chh))
        d.text((x * S, (ytop + chh + 8) * S), lab, font=f_s, fill=MUTED)
        x += cw + gap

    d.text((14 * S, 300 * S), "Stage 3  Annotate", font=f_b, fill=GOLD)
    d.text((14 * S, 340 * S),
           "Two people boxing the same person differently is a bug in the "
           "dataset.", font=f_l, fill=MUTED)

    rd = det(AI + "bus.jpg", conf=0.5, verbose=False)[0]
    boxes = [(b, rd.names[int(c)]) for b, c in zip(rd.boxes.xyxy, rd.boxes.cls)]
    person = [b for b, n in boxes if n == "person"]
    b = max(person, key=lambda q: (q[3] - q[1]) * (q[2] - q[0]))
    x1, y1, x2, y2 = [int(v) for v in b]
    pad = 26
    cx1, cy1 = max(x1 - pad, 0), max(y1 - pad, 0)
    cx2, cy2 = min(x2 + pad, w), min(y2 + pad, h)
    crop = base[cy1:cy2, cx1:cx2].copy()
    ox1, oy1, ox2, oy2 = x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1

    tight = crop.copy()
    cv2.rectangle(tight, (ox1, oy1), (ox2, oy2), C_PERSON, 6)
    loose = crop.copy()
    cv2.rectangle(loose, (max(ox1 - 20, 0), max(oy1 - 20, 0)),
                  (min(ox2 + 20, crop.shape[1]), min(oy2 + 20, crop.shape[0])),
                  C_BUS, 6)
    missed = crop.copy()

    cw2 = (W - 2 * PAD - 2 * gap) // 3
    ytop2, chh2 = 386, 198
    x = PAD
    for fr, lab, sub in [(tight, "Tight to the edge", "the rule most teams agree on"),
                         (loose, "Loose, drifting", "a tired labeller, late in the batch"),
                         (missed, "Not labelled", "taught to the model as background")]:
        bx, by, bw_, bh_ = paste_fit(img, d, fr, (x, ytop2, cw2, chh2))
        tx = (bx + bw_) // S + 20
        d.text((tx * S, (ytop2 + 40) * S), lab, font=f_s, fill=WHITE)
        for i, seg_ in enumerate(wrap(d, sub, f_s, (x + cw2 - tx) * S)):
            d.text((tx * S, (ytop2 + 78 + i * 32) * S), seg_, font=f_s,
                   fill=MUTED)
        x += cw2 + gap

    d.text((PAD * S, (H - 40) * S),
           "The same person, boxed three ways. Write the rule down before "
           "anyone starts, or the dataset records the disagreement.",
           font=f_n, fill=MUTED)
    img.save(OUT + "ch2_data.png")
    print("ch2_data.png")


# ==========================================================================
def fig_errors(det):
    base = cv2.imread(AI + "bus.jpg")
    hi = det(AI + "bus.jpg", conf=0.5, verbose=False)[0]
    lo = det(AI + "bus.jpg", conf=0.25, verbose=False)[0]

    n_hi = len(hi.boxes.cls)
    n_lo = len(lo.boxes.cls)
    lo_names = [lo.names[int(c)] for c in lo.boxes.cls]
    n_false = sum(1 for n in lo_names if n not in ("person", "bus"))
    n_true_lo = n_lo - n_false
    missed_hi = n_true_lo - n_hi

    a = draw_boxes(base, hi, colour_for, label=True)
    b = draw_boxes(base, lo, colour_for, label=True)

    img, d = canvas()
    f_l = font("Kanit", 29, "bold")
    f_s = font("Kanit", 25)
    f_n = font("Kanit", 25)
    PAD, gap = 14, 34
    cw = (W - 2 * PAD - 2 * gap) // 3
    top, chh = 52, 396

    tally = card(cw, chh - 16,
                 [("Found at 0.50", n_hi, rgb(C_PERSON)),
                  ("Found at 0.25", n_lo, rgb(C_PERSON)),
                  ("Of those, wrong", n_false, rgb(C_FALSE)),
                  ("Missed at 0.50", missed_hi, rgb(C_FALSE))],
                 ["Precision: of what it flagged, how much was real.",
                  "Recall: of what was there, how much it caught."])

    cells = [(a, "Threshold 0.50", "%d found, every one correct" % n_hi),
             (b, "Threshold 0.25", "%d found - the extra person is real, the "
                                   "stop sign is not" % n_lo),
             (tally, "What that costs", "same photo, same weights")]
    x = PAD
    for fr, lab, sub in cells:
        d.text((x * S, (top - 40) * S), lab, font=f_l, fill=GOLD)
        paste_fit(img, d, fr, (x, top, cw, chh))
        for i, seg_ in enumerate(wrap(d, sub, f_s, cw * S)):
            d.text((x * S, (top + chh + 12 + i * 32) * S), seg_, font=f_s,
                   fill=MUTED)
        x += cw + gap

    d.text((PAD * S, (H - 40) * S),
           "Lowering the threshold found the person half out of frame and "
           "invented a stop sign on the wall. Which error costs more is a "
           "business decision.", font=f_n, fill=MUTED)
    img.save(OUT + "ch2_errors.png")
    print("ch2_errors.png  hi=%d lo=%d false=%d missed=%d"
          % (n_hi, n_lo, n_false, missed_hi))


# ==========================================================================
CHAIN = [("PyTorch", ".pt from training"),
         ("ONNX", ".onnx, portable"),
         ("Runtime engine", "TensorRT / TFLite / OpenVINO / Hailo"),
         ("Edge device", "running on the line")]


def fig_field(det):
    base = cv2.imread(AI + "bus.jpg")
    h, w = base.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 13, 1.0)
    M[0, 2] += 70
    M[1, 2] += 40
    moved = cv2.warpAffine(base, M, (w, h), borderValue=(18, 18, 18))
    blur = cv2.GaussianBlur(base, (0, 0), 6)
    dusk = cv2.convertScaleAbs(base, alpha=0.38, beta=-12)

    def shot(im):
        r = det(im, conf=0.5, verbose=False)[0]
        names = [r.names[int(c)] for c in r.boxes.cls]
        return draw_boxes(im, r, colour_for, label=False, thick=5), Counter(names)

    img, d = canvas()
    f_c = font("Kanit", 29, "bold")
    f_cs = font("Kanit", 22)
    f_l = font("Kanit", 27, "bold")
    f_s = font("Kanit", 24)
    f_n = font("Kanit", 25)

    PAD, gap, arrow = 14, 12, 26
    n = len(CHAIN)
    bw = (W - 2 * PAD - (n - 1) * (gap + arrow)) // n
    bh, top = 112, 26
    x = PAD
    for i, (name, sub) in enumerate(CHAIN):
        d.rounded_rectangle([x * S, top * S, (x + bw) * S, (top + bh) * S],
                            radius=10 * S, fill=PANEL, outline=EDGE, width=2)
        d.text(((x + 20) * S, (top + 18) * S), name, font=f_c,
               fill=GOLD if i == n - 1 else WHITE)
        yy = top + 56
        for seg_ in wrap(d, sub, f_cs, (bw - 40) * S):
            d.text(((x + 20) * S, yy * S), seg_, font=f_cs, fill=MUTED)
            yy += 27
        if i < n - 1:
            ax = (x + bw + gap // 2 + 2) * S
            ay = (top + bh // 2) * S
            d.polygon([(ax, ay - 12 * S), (ax + 22 * S, ay),
                       (ax, ay + 12 * S)], fill=GOLD)
        x += bw + gap + arrow
    d.text((PAD * S, (top + bh + 10) * S),
           "Re-test after every conversion. Each step can change the output.",
           font=f_s, fill=GOLD)

    shots = [(base, "As trained on"), (moved, "Camera bumped"),
             (blur, "Motion blur"), (dusk, "Dusk")]
    cw2 = (W - 2 * PAD - 3 * 26) // 4
    ytop, chh = 230, 258
    x = PAD
    results = []
    for im, lab in shots:
        drawn, cnt = shot(im)
        results.append((lab, cnt))
        d.text((x * S, (ytop - 32) * S), lab, font=f_l, fill=WHITE)
        paste_fit(img, d, drawn, (x, ytop, cw2, chh))
        total = sum(cnt.values())
        line = "%d found  ·  %s" % (total, ", ".join(
            "%s %d" % (k, v) for k, v in sorted(cnt.items())) or "nothing")
        for i, seg_ in enumerate(wrap(d, line, f_s, cw2 * S)):
            d.text((x * S, (ytop + chh + 10 + i * 30) * S), seg_, font=f_s,
                   fill=GOLD if total < 4 else MUTED)
        x += cw2 + 26
    print("   field:", results)

    note = ("Same weights, same threshold, one altered photo each. Dusk did "
            "not break it; the camera moving lost a person and motion blur "
            "lost the bus. Log every result and save the frames it was "
            "unsure about.")
    ny = H - 66
    for ln in wrap(d, note, f_n, (W - 2 * PAD) * S):
        d.text((PAD * S, ny * S), ln, font=f_n, fill=MUTED)
        ny += 34
    img.save(OUT + "ch2_field.png")
    print("ch2_field.png")


if __name__ == "__main__":
    from ultralytics import YOLO
    det = YOLO("yolov8n.pt")
    seg = YOLO("yolov8n-seg.pt")
    cls = YOLO("yolov8n-cls.pt")
    fig_tasks(det, seg, cls)
    fig_data(det)
    fig_errors(det)
    fig_field(det)
