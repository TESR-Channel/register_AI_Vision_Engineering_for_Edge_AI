"""Chapter 4 figures built from real screenshots of the TESR Web Trainer.

The chapter described the browser steps in words. These are the steps.

The screenshots come from the actual repo checkout of AI_vision_Trainer served
locally, driven by Playwright with Chrome's fake capture device fed a real
photograph - so the camera view, the captured frames, the thumbnails and the
hand-drawn boxes are the page's own output, not a mock-up. The capture script
is slide_assets/src/shoot_trainer.py.

One honest limit: this machine cannot reach the MobileNet weights on tfhub, and
the page refuses to capture without a backbone, so the request for it was
answered with a stand-in of the right shape. That unlocks collect, label and
dataset export, all of which run the page's own code on real frames. It does
NOT make the Train or Live Test screens mean anything, so no figure here shows
a trained result.

Layout: a narrow left gutter of numbered notes, then the screenshot as large as
the figure box allows. No footnote inside the figure - the slide's own gold
caption does that job, and the screenshot needs the room.

Output: /home/claude/deck/img/ch4_web_*.png, for a 1920x1080 slide at 2x.
"""
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, "/home/claude/deck")
from ch0_panels import font, wrap, OUT, BG, EDGE, GOLD, MUTED, WHITE, S, W, H

SHOTS = "/tmp/shots/"
GUT = 300                      # left gutter for the numbered notes


def canvas():
    img = Image.new("RGB", (W * S, H * S), BG)
    return img, ImageDraw.Draw(img)


def place(img, d, name, box, crop=None, border=EDGE):
    """Drop a screenshot into a logical box, scaled to fit. Returns geometry."""
    x, y, bw, bh = box
    im = Image.open(SHOTS + name).convert("RGB")
    if crop:
        im = im.crop(crop)
    sc = min(bw * S / im.width, bh * S / im.height)
    tw, th = int(im.width * sc), int(im.height * sc)
    im = im.resize((tw, th), Image.LANCZOS)
    px = x * S + (bw * S - tw) // 2
    py = y * S + (bh * S - th) // 2
    img.paste(im, (px, py))
    d.rectangle([px - 2, py - 2, px + tw + 1, py + th + 1], outline=border,
                width=2)
    return (px / S, py / S, tw / S, th / S)


def disc(d, n, cx, cy, fnum, r=19):
    d.ellipse([(cx - r) * S, (cy - r) * S, (cx + r) * S, (cy + r) * S],
              fill=GOLD, outline=(12, 15, 19), width=4)
    wn = d.textlength(str(n), font=fnum)
    d.text((cx * S - wn / 2, (cy - 14) * S), str(n), font=fnum,
           fill=(14, 17, 22))


def mark(d, geom, fx, fy, n, fnum, r=19):
    """A numbered disc pinned to a point inside a placed screenshot."""
    px, py, tw, th = geom
    disc(d, n, px + tw * fx, py + th * fy, fnum, r=r)


def notes(d, items, fhead, fbody, fnum, top=46, x=14, width=GUT - 28):
    """Numbered notes down the left gutter."""
    y = top
    for i, (head, body) in enumerate(items, 1):
        disc(d, i, x + 20, y + 14, fnum)
        d.text(((x + 50) * S, (y - 2) * S), head, font=fhead, fill=GOLD)
        y += 36
        if body:
            for seg in wrap(d, body, fbody, (width - 36) * S):
                d.text(((x + 50) * S, y * S), seg, font=fbody, fill=MUTED)
                y += 28
        y += 22
    return y


# --------------------------------------------------------------------------
# 1. set the job up - task, then classes
# --------------------------------------------------------------------------
def fig_setup():
    img, d = canvas()
    f_h = font("Kanit", 25, "bold")
    f_b = font("Kanit", 22)
    f_n = font("Kanit", 24, "bold")

    notes(d, [("Choose the task",
               "Detection draws boxes and reports where the object is. "
               "Classification just names the frame."),
              ("Add your classes",
               "Two at least. A third class of the empty scene makes "
               "\"nothing here\" far steadier.")],
          f_h, f_b, f_n, top=70)

    g1 = place(img, d, "01_task.png", (GUT, 26, W - GUT - 14, 268))
    mark(d, g1, 0.268, 0.49, 1, f_n)
    g2 = place(img, d, "02_classes.png", (GUT, 320, W - GUT - 14, 256),
               crop=(0, 380, 2840, 900))
    mark(d, g2, 0.322, 0.152, 2, f_n)

    img.save(OUT + "ch4_web_setup.png")
    print("ch4_web_setup.png")


# --------------------------------------------------------------------------
# 2. collect - hold the button, watch the photos pile up
# --------------------------------------------------------------------------
def fig_collect():
    img, d = canvas()
    f_h = font("Kanit", 25, "bold")
    f_b = font("Kanit", 22)
    f_n = font("Kanit", 24, "bold")

    notes(d, [("Pick the class", "Whichever chip is lit is where the next "
                                 "photos go."),
              ("Hold to Capture", "Keep moving the object while you hold - "
                                  "angle, distance, light."),
              ("They pile up here", "Counted per class, and each one is "
                                    "waiting for a box.")],
          f_h, f_b, f_n, top=52)

    g = place(img, d, "04_collect_after.png", (GUT, 14, W - GUT - 14, 592),
              crop=(0, 0, 2840, 1160))
    # discs sit in the empty space beside each control, never on top of it
    mark(d, g, 0.168, 0.855, 1, f_n)
    mark(d, g, 0.420, 0.950, 2, f_n)
    mark(d, g, 0.498, 0.418, 3, f_n)

    img.save(OUT + "ch4_web_collect.png")
    print("ch4_web_collect.png")


# --------------------------------------------------------------------------
# 3. label - drag a box, red turns green
# --------------------------------------------------------------------------
def fig_label():
    img, d = canvas()
    f_h = font("Kanit", 25, "bold")
    f_b = font("Kanit", 22)
    f_n = font("Kanit", 24, "bold")
    f_c = font("Kanit", 23, "bold")

    notes(d, [("Click a photo", "Or the Label photos button - it opens the "
                                "next one without a box."),
              ("Drag a box", "Handles resize, inside moves, scroll rotates "
                             "for a tilted part."),
              ("Enter, then next", "Red turns green as each photo gets "
                                   "its box.")],
          f_h, f_b, f_n, top=40)

    top, mh = 26, 452
    mw = int(mh * 1240 / 1392)
    g1 = place(img, d, "07_label_box.png", (GUT + 6, top, mw, mh))
    g2 = place(img, d, "08_label_person.png", (GUT + 16 + mw, top, mw, mh))
    mark(d, g1, 0.70, 0.235, 2, f_n)
    mark(d, g2, 0.16, 0.762, 3, f_n)

    # only the first few thumbnails of both classes, so one green against
    # several red is actually legible
    sx = GUT + 30 + 2 * mw
    g3 = place(img, d, "08c_classes_labeled.png",
               (sx, 96, W - 14 - sx, 330), crop=(40, 516, 1010, 1130))
    mark(d, g3, 0.265, 0.305, 1, f_n, r=16)   # a red one, so the green shows
    d.text((sx * S, (96 + 340) * S), "green = labelled",
           font=f_c, fill=(111, 191, 115))
    d.text((sx * S, (96 + 372) * S), "red = still waiting for a box",
           font=f_c, fill=(214, 92, 88))

    img.save(OUT + "ch4_web_label.png")
    print("ch4_web_label.png")


if __name__ == "__main__":
    fig_setup()
    fig_collect()
    fig_label()
