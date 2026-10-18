"""Chapter 6 - the two real paths from a trained model to a running device.

Every command and artefact here comes from TESR's own AI Vision Web Trainer
repository, not from memory: `yolo-trainer/README.md` section 2, "Deploy to the
edge device", and the `edge-runtime/` folder beside it. The table there reads:

    Raspberry Pi   on your computer: python export.py --target pi
                   on the device:    copy best_ncnn_model/ and run
                   fastest:          re-export with --imgsz 320

    Jetson         on your computer: nothing - just copy best.pt
                   on the device:    runs on the GPU as is
                   fastest:          TensorRT (build_trt.sh)

That asymmetry is the whole lesson of the page: the Pi needs a conversion step
and the Jetson does not, because the Jetson runs the same CUDA stack the model
was trained on. Learners who expect one universal "export" step are the ones
who get stuck.

Two lanes, same starting file, so the fork is visible at a glance.

Output: /home/claude/deck/img/ch6_edge_path.png, for a 1713.6x560 figure box.
"""
import os
import sys

sys.path.insert(0, "/home/claude/deck")
from ip_panels2 import fig, t, bx, OUT, GOLD, WHITE, MUTED, GREEN, EDGE, BG

W, H = 1714, 560

LANE_PX, BLOCK_PX, SUB_PX, NOTE_PX = 30, 27, 22, 24
MONO = "DejaVu Sans Mono"

START = (40, 236)           # x, width of the shared starting block
COL = [(330, 352), (722, 352), (1114, 400)]   # the three stages, both lanes
BH = 116
Y1, Y2 = 72, 352            # top of each lane's blocks


def arrow(a, x0, x1, y, c=MUTED, lw=2.6):
    a.annotate("", xy=(x1, y), xytext=(x0, y),
               arrowprops=dict(arrowstyle="-|>", color=c, lw=lw,
                               mutation_scale=20, shrinkA=0, shrinkB=0),
               zorder=9)


def block(a, x, w, y, title, sub, edge, title_c=WHITE, mono=False):
    bx(a, x, y, w, BH, fc="#1b2027", ec=edge, lw=2, r=10)
    t(a, x + w / 2, y + 42, title, px=BLOCK_PX, c=title_c, ha="center",
      w="bold", fam=MONO if mono else "Kanit")
    t(a, x + w / 2, y + 80, sub, px=SUB_PX, c=MUTED, ha="center")


def panel_edge_path():
    f, a = fig(W, H)

    # ---- the file both lanes start from, centred between them
    sx, sw = START
    bx(a, sx, 180, sw, 180, fc="#1d2128", ec=GOLD, lw=2, r=10)
    t(a, sx + sw / 2, 224, "best.pt", px=30, c=GOLD, ha="center", w="bold",
      fam=MONO)
    t(a, sx + sw / 2, 266, "what you trained", px=SUB_PX, c=MUTED, ha="center")
    t(a, sx + sw / 2, 302, "on your computer", px=SUB_PX, c=MUTED, ha="center")

    # ---- lane 1: the Pi, which needs the conversion
    t(a, COL[0][0], Y1 - 30, "Raspberry Pi 4 / 5", px=LANE_PX, c=WHITE,
      w="bold")
    arrow(a, sx + sw + 14, COL[0][0] - 14, Y1 + BH / 2)
    block(a, COL[0][0], COL[0][1], Y1, "export.py --target pi",
          "run this on the computer", EDGE, mono=True)
    arrow(a, COL[0][0] + COL[0][1] + 14, COL[1][0] - 14, Y1 + BH / 2)
    block(a, COL[1][0], COL[1][1], Y1, "best_ncnn_model/",
          "a folder, not a single file", EDGE, mono=True)
    arrow(a, COL[1][0] + COL[1][1] + 14, COL[2][0] - 14, Y1 + BH / 2)
    block(a, COL[2][0], COL[2][1], Y1, "The Pi runs it",
          "copy that folder across", GREEN, title_c=GREEN)
    t(a, COL[2][0] + COL[2][1], Y1 + BH + 32,
      "smaller and faster:  export again with --imgsz 320", px=SUB_PX, c=GOLD,
      ha="right")

    # ---- lane 2: the Jetson, which skips it
    t(a, COL[0][0], Y2 - 30, "Jetson Nano / Orin Nano", px=LANE_PX, c=WHITE,
      w="bold")
    arrow(a, sx + sw + 14, COL[1][0] - 14, Y2 + BH / 2)
    # the skipped stage is left open on purpose, with the reason in its place
    t(a, COL[0][0] + COL[0][1] / 2, Y2 + BH / 2 - 36, "no conversion step",
      px=BLOCK_PX, c=MUTED, ha="center")
    t(a, COL[0][0] + COL[0][1] / 2, Y2 + BH / 2 + 32, "nothing to run here",
      px=SUB_PX, c=MUTED, ha="center")
    block(a, COL[1][0], COL[1][1], Y2, "best.pt",
          "the very same file", EDGE, mono=True)
    arrow(a, COL[1][0] + COL[1][1] + 14, COL[2][0] - 14, Y2 + BH / 2)
    block(a, COL[2][0], COL[2][1], Y2, "The Jetson runs it",
          "on the GPU, same CUDA stack", GREEN, title_c=GREEN)
    t(a, COL[2][0] + COL[2][1], Y2 + BH + 32,
      "faster still:  build a TensorRT engine with build_trt.sh", px=SUB_PX,
      c=GOLD, ha="right")

    t(a, 40, 536,
      "The Pi needs a conversion step. The Jetson does not, because it runs "
      "the same CUDA stack the model was trained on.", px=NOTE_PX, c=WHITE)

    f.savefig(OUT + "ch6_edge_path.png", dpi=100, facecolor=BG)
    print("ch6_edge_path.png")
    print("  source: AI_vision_Trainer/yolo-trainer/README.md section 2")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    panel_edge_path()
