"""Stand-alone QR badges for the TESR web tools.

Chapter 3 bakes its QR into the chapter figure. For chapters 4 and 5 the
badge is a separate image so it can be dropped onto a slide at whatever size
reads from the back of the room, next to a hyperlinked button for anyone
reading the PDF on a laptop.

Each badge is verified by decoding the QR back out of the finished PNG.

Output: /home/claude/deck/img/qr_*.png
"""
import cv2
import segno
from PIL import Image, ImageDraw
from pyzbar.pyzbar import decode

from ch0_panels import font, OUT, GOLD, MUTED, WHITE

S = 2
BG = (26, 21, 18)          # the deck's gold-tinted callout fill
EDGE = GOLD

TOOLS = [
    ("qr_trainer", "https://tesr-channel.github.io/AI_vision_Trainer/",
     "Train it on the web", "tesr-channel.github.io", "/AI_vision_Trainer"),
    ("qr_lab", "https://tesr-channel.github.io/image_processing_lab/",
     "Try it on the web", "tesr-channel.github.io", "/image_processing_lab"),
    ("qr_roi", "https://tesr-channel.github.io/AI_ROI_Designer/",
     "Open ROI Studio", "tesr-channel.github.io", "/AI_ROI_Designer"),
]

W, H = 700, 232


def badge(name, url, head, host, path):
    img = Image.new("RGB", (W * S, H * S), (15, 18, 22))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([2, 2, W * S - 3, H * S - 3], radius=16 * S,
                        fill=BG, outline=EDGE, width=5)

    qp = "/tmp/%s_src.png" % name
    segno.make(url, error="m").save(qp, scale=16, border=2,
                                    dark="#0F1216", light="#FFFFFF")
    q = Image.open(qp).convert("RGB").resize((172 * S, 172 * S),
                                             Image.NEAREST)
    img.paste(q, (30 * S, 30 * S))
    d.rectangle([30 * S - 3, 30 * S - 3, (30 + 172) * S + 2,
                 (30 + 172) * S + 2], outline=EDGE, width=3)

    f_h = font("Kanit", 34, "bold")
    f_u = font("DejaVu Sans Mono", 22)
    f_s = font("Kanit", 20)
    x = 232
    d.text((x * S, 52 * S), head, font=f_h, fill=GOLD)
    d.text((x * S, 104 * S), host, font=f_u, fill=WHITE)
    d.text((x * S, 136 * S), path, font=f_u, fill=WHITE)
    d.text((x * S, 176 * S), "scan it, or tap the link on the slide",
           font=f_s, fill=MUTED)

    out = OUT + name + ".png"
    img.save(out)
    got = [z.data.decode() for z in decode(cv2.imread(out))]
    assert got == [url], "QR did not survive rendering: %r" % (got,)
    print("%-12s ok  %s" % (name, got[0]))


if __name__ == "__main__":
    for t in TOOLS:
        badge(*t)
