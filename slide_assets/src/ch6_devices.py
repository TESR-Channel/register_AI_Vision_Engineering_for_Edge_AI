"""Chapter 6 - the three boards this course deploys to, and what each needs.

Drawn rather than photographed: the boards are simple original graphics, so
the slide carries no product photography, and the shapes are only there to
tell the three cards apart at a glance. The facts beside them come from TESR's
own AI Vision Web Trainer repository (`yolo-trainer/README.md`), which is the
toolchain the learner actually runs.

One honesty point the course requires: the README's tested paths are the
Raspberry Pi 4/5 and the Jetson Orin Nano on JetPack 6 and 7. The older Jetson
Nano is included because people own them, but it is marked as the one you will
have to adapt versions for - it is not a tested path here, and saying so is
better than letting someone discover it in a workshop.

No speed figures appear anywhere: the course measures frame rates on the
customer's own hardware or not at all.

Output: /home/claude/deck/img/ch6_devices.png, for a 1713.6x620 figure box.
"""
import os
import sys

sys.path.insert(0, "/home/claude/deck")
from ip_panels2 import fig, t, bx, OUT, GOLD, WHITE, MUTED, GREEN, CRIM, \
    EDGE, BG

W, H = 1714, 620
CARD_W, GAP, MARGIN = 540, 32, 30
NAME_PX, TAG_PX, ROW_PX, NOTE_PX = 30, 22, 24, 23

PI_GREEN = "#1f6b4a"
JET_DARK = "#2b3340"


def board(a, cx, cy, w, h, body, accent, header=True, heatsink=False):
    """A simple original board graphic - enough to tell the cards apart."""
    bx(a, cx - w / 2, cy - h / 2, w, h, fc=body, ec=accent, lw=2, r=8)
    if header:                      # the 40-pin header along the top edge
        for i in range(20):
            x = cx - w / 2 + 22 + i * ((w - 48) / 20)
            bx(a, x, cy - h / 2 + 12, 7, 16, fc="#d8c48a", ec="none", r=1)
    if heatsink:                    # a finned block in the middle
        bx(a, cx - 48, cy - 26, 96, 60, fc="#59636f", ec="#7b8794", lw=1.5,
           r=4)
        for i in range(6):
            bx(a, cx - 42 + i * 15, cy - 20, 6, 48, fc="#7b8794", ec="none",
               r=1)
    else:                           # a couple of port blocks on the edge
        for i in range(2):
            bx(a, cx + w / 2 - 54, cy - 18 + i * 34, 48, 26, fc="#59636f",
               ec="none", r=3)
    bx(a, cx - w / 2 + 26, cy + h / 2 - 40, 46, 28, fc="#3b4552", ec="none",
       r=3)


DEVICES = [
    dict(name="Raspberry Pi 4 / 5", tag="tested path", tag_c=GREEN,
         body=PI_GREEN, accent="#8fd6b4", heatsink=False,
         rows=[("Processor", "CPU only"),
               ("Before you copy", "export.py --target pi"),
               ("What you copy", "best_ncnn_model/ — a folder"),
               ("On the device", "a venv, never sudo pip3"),
               ("If it is slow", "export again with --imgsz 320")]),
    dict(name="Jetson Orin Nano", tag="tested path", tag_c=GREEN,
         body=JET_DARK, accent="#9fb2c9", heatsink=True,
         rows=[("Processor", "CUDA GPU"),
               ("Before you copy", "nothing — no export"),
               ("What you copy", "best.pt — the same file"),
               ("On the device", "check JetPack first"),
               ("When the model is final", "build a TensorRT engine")]),
    dict(name="Jetson Nano (older)", tag="expect to adapt", tag_c=CRIM,
         body=JET_DARK, accent="#7b8794", heatsink=True,
         rows=[("Processor", "CUDA GPU, earlier generation"),
               ("Before you copy", "nothing — no export"),
               ("What you copy", "best.pt"),
               ("On the device", "an older JetPack"),
               ("Worth knowing", "not a tested path in this course")]),
]


def card(a, x, d):
    bx(a, x, 16, CARD_W, 556, fc="#161a20", ec=EDGE, lw=2, r=12)
    t(a, x + 26, 56, d["name"], px=NAME_PX, c=WHITE, w="bold")
    t(a, x + CARD_W - 26, 56, d["tag"], px=TAG_PX, c=d["tag_c"], ha="right",
      w="bold")
    board(a, x + CARD_W / 2, 158, 300, 152, d["body"], d["accent"],
          heatsink=d["heatsink"])
    y = 272
    for label, value in d["rows"]:
        t(a, x + 26, y, label, px=TAG_PX, c=GOLD)
        t(a, x + 26, y + 32, value, px=ROW_PX, c=WHITE)
        y += 62


def panel_devices():
    f, a = fig(W, H)
    for i, d in enumerate(DEVICES):
        card(a, MARGIN + i * (CARD_W + GAP), d)
    f.savefig(OUT + "ch6_devices.png", dpi=100, facecolor=BG)
    print("ch6_devices.png")
    print("  facts from AI_vision_Trainer/yolo-trainer/README.md")
    print("  no frame-rate figures, per the course rule")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    panel_devices()
