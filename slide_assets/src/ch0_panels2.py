"""More figures for chapter 0, the course overview.

Chapter 0 is the first thing anyone sees, so it carries the heaviest burden
of showing rather than telling. These four figures go on the pages that were
still text only.

Real OpenCV output is used wherever a result is shown. Where a figure is a
diagram - a device, a worked sum - it says so on the figure itself.

Output: /home/claude/deck/img/ch0_*.png, for a 1920x1080 slide at 2x.
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw

from ch0_panels import (font, wrap, OUT, BG, PANEL, EDGE, GOLD, MUTED, WHITE,
                        S, W, H)

SAMPLES = "/home/claude/image_processing_lab/samples/"
GREEN = (111, 191, 115)
BLUE = (214, 92, 88)


def canvas():
    img = Image.new("RGB", (W * S, H * S), BG)
    return img, ImageDraw.Draw(img)


def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else \
           cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def paste_fit(img, d, frame, box, outline=EDGE):
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


def red_discs():
    img = cv2.imread(SAMPLES + "ex3.png")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 120, 70), (4, 255, 255)) | \
        cv2.inRange(hsv, (172, 120, 70), (179, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return img, [c for c in cs if cv2.contourArea(c) > 400]


def device_card(cw, ch, lines, head):
    """A drawn edge device, labelled as a diagram - not a photo."""
    im = Image.new("RGB", (cw * S, ch * S), BG)
    d = ImageDraw.Draw(im)
    f_h = font("Kanit", 26, "bold")
    f_m = font("DejaVu Sans Mono", 19)

    bw_, bh_ = cw - 40, 150
    bx_, by_ = 20, 40
    d.rounded_rectangle([bx_ * S, by_ * S, (bx_ + bw_) * S, (by_ + bh_) * S],
                        radius=12 * S, fill=PANEL, outline=GOLD, width=3)
    d.text(((bx_ + 20) * S, (by_ + 18) * S), head, font=f_h, fill=GOLD)
    # a row of pins along the bottom edge, so it reads as a board
    for i in range(16):
        px = bx_ + 24 + i * 15
        d.rectangle([px * S, (by_ + bh_ - 12) * S, (px + 7) * S,
                     (by_ + bh_ - 4) * S], fill=EDGE)
    y = by_ + bh_ + 24
    for ln in lines:
        d.text((bx_ * S + 20 * S, y * S), ln, font=f_m, fill=MUTED)
        y += 30
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------
# ch0_arc - what the two days actually produce
# --------------------------------------------------------------------------
def fig_arc():
    src, cs = red_discs()

    raw = src.copy()

    boxed = src.copy()
    for c in cs:
        x, y, w_, h_ = cv2.boundingRect(c)
        cv2.rectangle(boxed, (x, y), (x + w_, y + h_), (88, 92, 214), 3)

    labelled = src.copy()
    for i, c in enumerate(sorted(cs, key=cv2.contourArea, reverse=True)[:3]):
        x, y, w_, h_ = cv2.boundingRect(c)
        cv2.rectangle(labelled, (x, y), (x + w_, y + h_), (115, 191, 111), 3)
        cv2.rectangle(labelled, (x, y - 26), (x + 118, y), (115, 191, 111), -1)
        cv2.putText(labelled, "part_a", (x + 6, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 24, 20), 2)

    img, d = canvas()
    f_l = font("Kanit", 28, "bold")
    f_s = font("Kanit", 24)
    f_n = font("Kanit", 25)

    PAD, gap, arrow = 14, 26, 34
    cw = (W - 2 * PAD - 3 * (gap + arrow)) // 4
    top, chh = 60, 436

    dev = device_card(cw, chh, ["$ python predict.py",
                                "part_a  x=412 y=236",
                                "mqtt  tesr/line1/count 6"],
                      "Raspberry Pi / Jetson")

    cells = [(raw, "A camera on the bench", "any USB webcam"),
             (boxed, "Image processing", "tuned on the web, code exported"),
             (labelled, "Your own images", "labelled, then trained"),
             (dev, "Running on the device", "diagram - result goes to MQTT")]

    x = PAD
    for i, (fr, lab, sub) in enumerate(cells):
        d.text((x * S, (top - 38) * S), lab, font=f_l,
               fill=GOLD if i == len(cells) - 1 else WHITE)
        paste_fit(img, d, fr, (x, top, cw, chh),
                  outline=BG if i == 3 else EDGE)
        d.text((x * S, (top + chh + 10) * S), sub, font=f_s, fill=MUTED)
        if i < len(cells) - 1:
            ax = (x + cw + gap // 2 + 4) * S
            ay = (top + chh // 2) * S
            d.polygon([(ax, ay - 14 * S), (ax + 26 * S, ay),
                       (ax, ay + 14 * S)], fill=GOLD)
        x += cw + gap + arrow

    d.text((PAD * S, (H - 40) * S),
           "Two days, end to end. The first three panels are real output from "
           "the course sample images; the fourth is a diagram of where it "
           "ends up.", font=f_n, fill=MUTED)
    img.save(OUT + "ch0_arc.png")
    print("ch0_arc.png")


# --------------------------------------------------------------------------
# ch0_takehome - the files that leave with the participant
# --------------------------------------------------------------------------
TREE = [
    ("workshop/", "", True),
    ("  best.pt", "the model you trained, in PyTorch", False),
    ("  model.onnx", "the portable export", False),
    ("  predict.py", "runs on PC, Raspberry Pi and Jetson", False),
    ("  install_pi.sh", "sets the device up", False),
    ("  build_trt.sh", "builds the TensorRT engine on Jetson", False),
    ("  README_DEPLOY.md", "how to run it, step by step", False),
    ("  dataset/", "your own photos and their labels", True),
    ("  notebooks/", "every workshop script, commented", True),
]


def fig_takehome():
    img, d = canvas()
    f_f = font("DejaVu Sans Mono", 26)
    f_c = font("Kanit", 24)
    f_h = font("Kanit", 30, "bold")
    f_n = font("Kanit", 25)

    PAD = 14
    cw, ch = W - 2 * PAD, 452
    top = 40
    d.rounded_rectangle([PAD * S, top * S, (PAD + cw) * S, (top + ch) * S],
                        radius=12 * S, fill=PANEL, outline=EDGE, width=2)

    d.text((60 * S, (top + 26) * S), "What leaves with you at the end of day two",
           font=f_h, fill=GOLD)

    y = top + 86
    for name, note, is_dir in TREE:
        d.text((60 * S, y * S), name, font=f_f,
               fill=GOLD if is_dir else WHITE)
        if note:
            d.text((700 * S, (y + 3) * S), note, font=f_c, fill=MUTED)
        y += 40

    d.text((PAD * S, (H - 50) * S),
           "Real file names from the export, not placeholders. Nothing here "
           "needs TESR afterwards - it runs on your own machine and your own "
           "device.", font=f_n, fill=MUTED)
    img.save(OUT + "ch0_takehome.png")
    print("ch0_takehome.png")


# --------------------------------------------------------------------------
# ch0_measure - pixels into millimetres, the worked example
# --------------------------------------------------------------------------
REF_MM = 10.0
REF_PX = 120.0
PART_PX = 480.0


def fig_measure():
    src = cv2.imread(SAMPLES + "ex3.png")
    h0, w0 = src.shape[:2]

    # panel 1 - a known length laid on the scene
    a = src.copy()
    y0 = int(h0 * 0.86)
    x0 = int(w0 * 0.10)
    cv2.line(a, (x0, y0), (x0 + 120, y0), (65, 164, 217), 4)
    for xx in (x0, x0 + 120):
        cv2.line(a, (xx, y0 - 12), (xx, y0 + 12), (65, 164, 217), 4)
    cv2.putText(a, "10 mm", (x0, y0 - 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (65, 164, 217), 2)

    # panel 2 - the same length counted in pixels
    b = src.copy()
    cv2.line(b, (x0, y0), (x0 + 120, y0), (115, 191, 111), 4)
    for xx in (x0, x0 + 120):
        cv2.line(b, (xx, y0 - 12), (xx, y0 + 12), (115, 191, 111), 4)
    cv2.putText(b, "120 px", (x0, y0 - 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (115, 191, 111), 2)

    img, d = canvas()
    f_l = font("Kanit", 28, "bold")
    f_s = font("Kanit", 24)
    f_n = font("Kanit", 25)
    f_big = font("DejaVu Sans Mono", 34, "bold")
    f_m = font("DejaVu Sans Mono", 24)

    PAD, gap = 14, 32
    cw = (W - 2 * PAD - 2 * gap) // 3
    top, chh = 56, 424

    cells = [(a, "Put a known length in the scene",
              "a ruler, a gauge block, a printed mark"),
             (b, "Count how many pixels it spans",
              "the camera does not move after this")]
    x = PAD
    for fr, lab, sub in cells:
        d.text((x * S, (top - 38) * S), lab, font=f_l, fill=WHITE)
        paste_fit(img, d, fr, (x, top, cw, chh))
        d.text((x * S, (top + chh + 10) * S), sub, font=f_s, fill=MUTED)
        x += cw + gap

    # the sum
    d.text((x * S, (top - 38) * S), "Now every pixel has a size",
           font=f_l, fill=GOLD)
    d.rounded_rectangle([x * S, top * S, (x + cw) * S, (top + chh) * S],
                        radius=12 * S, fill=PANEL, outline=EDGE, width=2)
    mm_per_px = REF_MM / REF_PX
    part_mm = PART_PX * mm_per_px
    M = 30
    yy = top + 44
    for line, col, fnt in [
            ("%.0f mm / %.0f px" % (REF_MM, REF_PX), MUTED, f_m),
            ("= %.4f mm per pixel" % mm_per_px, GOLD, f_big),
            ("", MUTED, f_m),
            ("a part spanning %.0f px" % PART_PX, MUTED, f_m),
            ("= %.1f mm" % part_mm, GOLD, f_big)]:
        if line:
            d.text(((x + M) * S, yy * S), line, font=fnt, fill=col)
        yy += 46 if fnt is f_big else 38

    for i, ln in enumerate(wrap(d,
                                "Re-measure the reference whenever the camera "
                                "or the lens moves. The scale belongs to that "
                                "one setup, not to the camera.",
                                f_s, (cw - 2 * M) * S)):
        d.text(((x + M) * S, (yy + 14 + i * 32) * S), ln, font=f_s, fill=MUTED)

    note = ("A worked example with chosen numbers, to show the method - not "
            "a measurement of this photo. Accuracy depends on the lens, the "
            "distance and how square the part sits.")
    ny = H - 76
    for ln in wrap(d, note, f_n, (W - 2 * PAD) * S):
        d.text((PAD * S, ny * S), ln, font=f_n, fill=MUTED)
        ny += 34
    img.save(OUT + "ch0_measure.png")
    print("ch0_measure.png  %.4f mm/px  %.1f mm" % (mm_per_px, part_mm))


if __name__ == "__main__":
    fig_arc()
    fig_takehome()
    fig_measure()
