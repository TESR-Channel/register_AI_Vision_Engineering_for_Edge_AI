"""Chapter 4 page 2 - rules you write, against examples you show.

The page used to be an abstract flow diagram: Image -> Rules you wrote ->
Answer, over labelled examples -> Training -> Model -> Answer. True, but it
asks the reader to imagine what each box means.

This does it on one photograph instead. The same frame, the same lamp
change, two approaches:

  colour rule you wrote   as photographed  2 blobs, 36,016 px
                          lamp dimmed      0 blobs, nothing at all
  model                   as photographed  4 objects - bus, person
                          lamp dimmed      4 objects - unchanged

Measured at render time, printed to stdout when this runs. If the numbers
move, the figure moves with them - nothing here is typed in by hand.

Output: /home/claude/deck/img/ch4_rules.png, for a 1920x1080 slide at 2x.
"""
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
C_PERSON = (115, 191, 111)
C_BUS = (88, 92, 214)

# the six numbers a person has to choose for the colour rule
LO, HI = (95, 90, 70), (125, 255, 255)


def band(im):
    h = im.shape[0]
    return im[int(h * 0.27):int(h * 0.81), :]


def colour_rule(img):
    """Find the bus by colour, the way a person would write it by hand."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, LO, HI)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs = [c for c in cs if cv2.contourArea(c) > 8000]
    out = img.copy()
    tint = np.zeros_like(img)
    tint[:] = (217, 164, 65)
    mask = m.astype(bool)
    out = np.where(mask[:, :, None], cv2.addWeighted(img, 0.35, tint, 0.65, 0),
                   out)
    for c in cs:
        x, y, bw, bh = cv2.boundingRect(c)
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (65, 164, 217), 4)
    return out, len(cs)


def model_run(det, img):
    r = det(img, conf=0.5, verbose=False)[0]
    out = img.copy()
    for b, c in zip(r.boxes.xyxy, r.boxes.cls):
        x1, y1, x2, y2 = [int(v) for v in b]
        nm = r.names[int(c)]
        col = C_PERSON if nm == "person" else C_BUS
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 4)
    names = [r.names[int(c)] for c in r.boxes.cls]
    return out, len(names), Counter(names)


def paste_fit(img, d, frame, box, outline=EDGE):
    x, y, bw, bh = box
    a = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ih, iw = a.shape[:2]
    sc = min(bw / iw, bh / ih)
    tw, th = int(iw * sc), int(ih * sc)
    tile = Image.fromarray(cv2.resize(a, (tw * S, th * S),
                                      interpolation=cv2.INTER_AREA))
    px = (x + (bw - tw) // 2) * S
    py = (y + (bh - th) // 2) * S
    img.paste(tile, (px, py))
    d.rectangle([px - 2, py - 2, px + tw * S + 1, py + th * S + 1],
                outline=outline, width=2)


def build():
    from ultralytics import YOLO
    det = YOLO("yolov8n.pt")

    base = cv2.imread(AI + "bus.jpg")
    dusk = cv2.convertScaleAbs(base, alpha=0.38, beta=-12)

    rule_ok, n_rule_ok = colour_rule(base)
    rule_bad, n_rule_bad = colour_rule(dusk)
    ml_ok, n_ml_ok, c_ok = model_run(det, base)
    ml_bad, n_ml_bad, c_bad = model_run(det, dusk)
    print("rule : %d -> %d    model: %d %s -> %d %s"
          % (n_rule_ok, n_rule_bad, n_ml_ok, dict(c_ok), n_ml_bad, dict(c_bad)))

    img = Image.new("RGB", (W * S, H * S), BG)
    d = ImageDraw.Draw(img)
    f_row = font("Kanit", 29, "bold")
    f_sub = font("Kanit", 22)
    f_code = font("DejaVu Sans Mono", 19)
    f_col = font("Kanit", 23, "bold")
    f_res = font("Kanit", 27, "bold")

    PAD = 14
    lab_w = 392
    cell_w = (W - 2 * PAD - lab_w - 3 * 18) // 2
    rows_top = [86, 338]
    col_x = [PAD + lab_w + 18, PAD + lab_w + 18 * 2 + cell_w]

    d.text((col_x[0] * S, 52 * S), "as photographed", font=f_col, fill=MUTED)
    d.text((col_x[1] * S, 52 * S), "same frame, lamp dimmed", font=f_col,
           fill=MUTED)

    # ---------- row 1: the rule you write ----------
    ry = rows_top[0]
    d.text((PAD * S, ry * S), "You write the rule", font=f_row, fill=GOLD)
    d.text((PAD * S, (ry + 38) * S), "you choose all six numbers yourself",
           font=f_sub, fill=MUTED)
    code = ["cv2.inRange(hsv,", "   (95, 90, 70),", "   (125,255,255))"]
    for i, ln in enumerate(code):
        d.text((PAD * S, (ry + 80 + i * 28) * S), ln, font=f_code, fill=WHITE)

    for i, (frame, n) in enumerate([(rule_ok, n_rule_ok),
                                    (rule_bad, n_rule_bad)]):
        paste_fit(img, d, band(frame), (col_x[i], ry, cell_w, 178))
        col = GREEN if n else RED
        txt = ("%d blue regions found" % n) if n else "nothing found at all"
        d.text((col_x[i] * S, (ry + 190) * S), txt, font=f_res, fill=col)

    # ---------- row 2: the examples you show ----------
    ry = rows_top[1]
    d.text((PAD * S, ry * S), "You show it examples", font=f_row, fill=GOLD)
    d.text((PAD * S, (ry + 38) * S), "nobody picks a number at all",
           font=f_sub, fill=MUTED)
    for i, ln in enumerate(["labelled photos  ->", "   training  ->",
                            "   a model"]):
        d.text((PAD * S, (ry + 80 + i * 28) * S), ln, font=f_code, fill=WHITE)

    for i, (frame, n, cnt) in enumerate([(ml_ok, n_ml_ok, c_ok),
                                         (ml_bad, n_ml_bad, c_bad)]):
        paste_fit(img, d, band(frame), (col_x[i], ry, cell_w, 178))
        d.text((col_x[i] * S, (ry + 190) * S),
               "%d objects found" % n, font=f_res, fill=GREEN)

    note = ("Measured on this photograph when the slide was built. The rule "
            "you wrote stops working when the lamp changes; the model never "
            "noticed. That is the whole trade - you pay for it in labelled "
            "examples.")
    ny = H - 64
    for ln in wrap(d, note, f_sub, (W - 2 * PAD) * S):
        d.text((PAD * S, ny * S), ln, font=f_sub, fill=MUTED)
        ny += 30

    img.save(OUT + "ch4_rules.png")
    print("ch4_rules.png")


if __name__ == "__main__":
    build()
