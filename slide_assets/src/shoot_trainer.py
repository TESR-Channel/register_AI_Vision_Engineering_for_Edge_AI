"""Screenshot the real TESR AI Vision Web Trainer UI, step by step.

The page is the actual repo checkout served locally, with its CDN libraries
vendored from npm so nothing has to be fetched from the open internet. The
webcam is Chrome's fake capture device fed a real photograph, so the frames the
page captures are genuine frames from a genuine camera stream.

The MobileNet backbone downloads its weights from tfhub, which this machine
cannot reach. The page refuses to capture a photo without a backbone, so the
request for it is answered with a stand-in that returns a cheap pixel-derived
embedding of the right shape. That unlocks the parts of the tool these
screenshots are for - choose a task, add classes, capture from the camera, draw
a box, export the dataset - all of which run the page's own code on real camera
frames.

What it does NOT unlock is anything that depends on the embedding meaning
something: Train and Live Test. No screenshot here shows a trained result, and
none should.

Output: /tmp/shots/*.png
"""
import pathlib
import sys
import time

from playwright.sync_api import sync_playwright

OUT = pathlib.Path("/tmp/shots")
OUT.mkdir(exist_ok=True)
URL = "http://127.0.0.1:8099/index.html"
CLASSES = ["bus", "person"]

FLAGS = [
    "--use-fake-ui-for-media-stream",
    "--use-fake-device-for-media-stream",
    "--use-file-for-fake-video-capture=/tmp/fakecam.y4m",
    "--allow-file-access-from-files",
    "--autoplay-policy=no-user-gesture-required",
]


def shot(page, name, selector=None, pad=0):
    path = OUT / ("%s.png" % name)
    if selector:
        el = page.query_selector(selector)
        if el is None:
            print("  !! missing selector", selector)
            return
        el.scroll_into_view_if_needed()
        page.wait_for_timeout(350)
        el.screenshot(path=str(path))
    else:
        page.screenshot(path=str(path))
    print("  ->", path.name)


SHIM = """
window.mobilenet = {
  load: async () => ({
    infer: (el) => tf.tidy(() => {
      const img = tf.browser.fromPixels(el).toFloat().div(255);
      const small = tf.image.resizeBilinear(img, [20, 20]).reshape([1, 1200]);
      return tf.concat([small, tf.zeros([1, 80])], 1);
    })
  })
};
"""


def main():
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=FLAGS)
        ctx = browser.new_context(
            viewport={"width": 1500, "height": 1000},
            device_scale_factor=2,
            permissions=["camera"],
            locale="en-US",
        )
        page = ctx.new_page()
        page.route("**/vendor/mobilenet.min.js",
                   lambda route: route.fulfill(
                       status=200, content_type="application/javascript",
                       body=SHIM))
        page.on("console", lambda m: print("   [console]", m.type, m.text[:160]))
        page.on("pageerror", lambda e: print("   [pageerror]", str(e)[:200]))
        page.goto(URL, wait_until="load")
        page.wait_for_timeout(4000)

        print("title:", page.title())
        shot(page, "00_landing")

        # ---- 1. task ----
        page.click("#taskDetect")
        page.wait_for_timeout(600)
        shot(page, "01_task", "#taskCard")

        # ---- 2. classes ----
        for c in CLASSES:
            page.fill("#newCls", c)
            page.click("text=+ Add Class")
            page.wait_for_timeout(500)
        shot(page, "02_classes", "#clsCard")

        # ---- 3. collect ----
        page.wait_for_timeout(2500)          # let the stream settle
        video_ok = page.evaluate(
            "() => { const v = document.getElementById('cam');"
            " return v ? [v.videoWidth, v.videoHeight, v.readyState] : null; }")
        print("  video:", video_ok)
        shot(page, "03_collect_before", "#camCard")

        # Hold the real button first, then top up through the page's own
        # addExample - the burst fires every 160ms but each capture has to wait
        # for an embedding, and on a CPU-only backend only one lands per hold.
        # Same code path, same moving camera, just driven at a pace this
        # machine can keep up with.
        for i, c in enumerate(CLASSES):
            chips = page.query_selector_all("#capClsChips .clschip")
            print("  chips:", len(chips))
            if i < len(chips):
                chips[i].click()
                page.wait_for_timeout(400)
            btn = page.query_selector("#holdBtn")
            box = btn.bounding_box()
            page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            page.mouse.down()
            page.wait_for_timeout(1200)
            page.mouse.up()
            page.wait_for_timeout(400)
            page.evaluate(
                """async (cls) => {
                     const v = document.getElementById('cam');
                     for (let i = 0; i < 17; i++) {
                       await addExample(cls, v, snapThumb(v));
                       await new Promise(r => setTimeout(r, 90));
                     }
                   }""", c)
            page.wait_for_timeout(600)
            print("  captured for", c, "->", page.inner_text("#capMsg")[:80])

        shot(page, "04_collect_after", "#camCard")
        shot(page, "05_classes_filled", "#clsCard")

        # ---- 4. labelling ----
        lbl = page.query_selector("#btnLabel")
        print("  label button visible:", lbl.is_visible() if lbl else None)
        if lbl and lbl.is_visible():
            lbl.click()
        else:
            page.evaluate("openNextUnlabeled()")
        page.wait_for_timeout(1200)
        shot(page, "06_label_empty", "#lblModal .mbox")

        def drag_box(x0, y0, x1, y1):
            cv = page.query_selector("#lblCv")
            b = cv.bounding_box()
            page.mouse.move(b["x"] + b["width"] * x0, b["y"] + b["height"] * y0)
            page.mouse.down()
            for t in range(1, 13):
                page.mouse.move(b["x"] + b["width"] * (x0 + (x1 - x0) * t / 12),
                                b["y"] + b["height"] * (y0 + (y1 - y0) * t / 12))
                page.wait_for_timeout(35)
            page.mouse.up()
            page.wait_for_timeout(700)

        # the bus fills the frame; the box is drawn to its edges
        drag_box(0.045, 0.165, 0.965, 0.745)
        shot(page, "07_label_box", "#lblModal .mbox")

        page.keyboard.press("Enter")
        page.wait_for_timeout(900)
        page.keyboard.press("Escape")
        page.wait_for_timeout(500)

        # open a photo from the other class by clicking its thumbnail, the way
        # the card tells you to ("click to label"), and box the person
        thumbs = page.query_selector_all("#clsList > div")
        print("  class blocks:", len(thumbs))
        person_thumb = page.query_selector_all("#clsList > div:nth-child(2) img")
        print("  person thumbs:", len(person_thumb))
        if person_thumb:
            person_thumb[0].click()
            page.wait_for_timeout(1000)
            drag_box(0.085, 0.265, 0.315, 0.960)
            shot(page, "08_label_person", "#lblModal .mbox")
            page.keyboard.press("Enter")
            page.wait_for_timeout(800)
            page.keyboard.press("Escape")
            page.wait_for_timeout(600)
        shot(page, "08c_classes_labeled", "#clsCard")

        # ---- 5. dataset / augmentation / train card ----
        shot(page, "09_dataset", "#camCard")
        shot(page, "10_augment", "#augCard")
        shot(page, "11_train", "#trainCard")
        shot(page, "12_steps", ".steps")

        # the test/export card only appears after a successful train; show the
        # real panel so the slide can point at where the buttons live
        page.evaluate("document.getElementById('testCard').style.display='block'")
        page.wait_for_timeout(500)
        shot(page, "13_export", "#testCard")

        page.wait_for_timeout(300)
        shot(page, "99_full")
        browser.close()


if __name__ == "__main__":
    sys.exit(main())
