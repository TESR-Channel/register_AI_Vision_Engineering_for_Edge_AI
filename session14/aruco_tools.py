#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Session 14 - ArUco markers: a known size in the frame
=====================================================
TESR Academy - AI Vision Engineering for Edge AI

A camera measures in pixels. A pixel is not a millimetre until something of a
known size is in the picture. An ArUco marker is that something: it is cheap to
print, the detector finds it reliably, and because you know how wide you printed
it, every other distance in that plane follows.

Four things, one file:

    generate    print a sheet of markers at an exact millimetre size
    detect      find markers in an image or a camera and report their IDs
    scale       work out the pixels-per-millimetre and write the ROI Studio
                "scale" block for you
    rectify     flatten the perspective using four corner markers

Minimum requirement : Python 3.8+, opencv-contrib-python, numpy
                      (plain opencv-python does NOT include cv2.aruco)

Usage
-----
    python aruco_tools.py generate --ids 0 1 2 3 --mm 40 --out markers/
    python aruco_tools.py detect   --image frame.jpg
    python aruco_tools.py detect   --camera 0
    python aruco_tools.py scale    --image frame.jpg --mm 40
    python aruco_tools.py rectify  --image frame.jpg --ids 0 1 2 3 --mm 40 \
                                   --size 190 277 --out flat.jpg

--size is measured corner to corner between the markers, not the paper size.
Lay the sheet out, measure outer corner to outer corner with a tape, and pass
that. rectify checks itself afterwards and tells you if the number was wrong.

Author: TESR Co., Ltd. - Thai Embedded Systems and Robotics
License: MIT
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

# 4x4 is the right default for a workshop: the cells are large, so it survives
# a cheap printer, a wet lens and a camera further back than you planned. Larger
# dictionaries hold more IDs but need more pixels on target to decode.
DEFAULT_DICT = "DICT_4X4_50"

OK, WARN, FAIL = "[PASS]", "[WARN]", "[FAIL]"


def get_dict(name):
    const = getattr(cv2.aruco, name, None)
    if const is None:
        sys.exit("%s unknown dictionary: %s" % (FAIL, name))
    return cv2.aruco.getPredefinedDictionary(const)


def make_detector(name):
    """Works across the OpenCV 4.7 API change and later."""
    d = get_dict(name)
    params = cv2.aruco.DetectorParameters()
    # Sub-pixel corners. Without this the scale wobbles by a pixel or so between
    # frames, which is the difference between 40.0 mm and 40.4 mm on a 400 px
    # marker - small, but it is the whole measurement.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(d, params)


def detect_markers(img, dict_name=DEFAULT_DICT):
    """-> {id: corners(4,2) float32} in the order the detector returns them."""
    det = make_detector(dict_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    corners, ids, _rej = det.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return {}
    return {int(i): c.reshape(4, 2).astype(np.float32)
            for i, c in zip(ids.flatten(), corners)}


def marker_px(c):
    """Side length in pixels, averaged over all four sides.

    Averaging matters: a marker seen at an angle has a short side and a long
    one, and the average is much closer to the truth than any single edge.
    """
    return float(np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]))


# --------------------------------------------------------------------- generate
def cmd_generate(a):
    os.makedirs(a.out, exist_ok=True)
    d = get_dict(a.dict)
    # 8 px per mm -> a 40 mm marker is 320 px, which prints cleanly at 200 dpi
    # and leaves room for the quiet zone. Printing smaller than the marker is
    # the usual reason a detector "does not work".
    px_per_mm = 8
    side = int(round(a.mm * px_per_mm))
    quiet = max(8, side // 8)          # the white border is part of the marker

    for mid in a.ids:
        img = cv2.aruco.generateImageMarker(d, mid, side)
        sheet = np.full((side + quiet * 2, side + quiet * 2), 255, np.uint8)
        sheet[quiet:quiet + side, quiet:quiet + side] = img
        sheet = cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR)
        cv2.putText(sheet, "id=%d  %gmm  %s" % (mid, a.mm, a.dict),
                    (quiet, sheet.shape[0] - quiet // 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1, cv2.LINE_AA)
        path = os.path.join(a.out, "aruco_%s_id%02d_%gmm.png" % (a.dict, mid, a.mm))
        cv2.imwrite(path, sheet)
        print("%s wrote %s" % (OK, path))

    print()
    print("Print these at 100% scale - NOT 'fit to page', which silently resizes")
    print("them and makes every measurement wrong. Then measure one with a ruler")
    print("and confirm it really is %g mm before you trust anything downstream." % a.mm)
    return 0


# ----------------------------------------------------------------------- detect
def cmd_detect(a):
    if a.image:
        img = cv2.imread(a.image)
        if img is None:
            print("%s cannot read %s" % (FAIL, a.image))
            return 1
        return report(img, a, show=a.show)

    cap = cv2.VideoCapture(int(a.camera))
    if not cap.isOpened():
        print("%s cannot open camera %s" % (FAIL, a.camera))
        return 1
    print("camera %s open - q or Ctrl-C to stop" % a.camera)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("%s dropped frame" % WARN)
                break
            found = detect_markers(frame, a.dict)
            vis = frame.copy()
            for mid, c in found.items():
                cv2.polylines(vis, [c.astype(int)], True, (65, 164, 217), 2)
                cv2.putText(vis, str(mid), tuple(c[0].astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (65, 164, 217), 2)
            print("\r%-60s" % ("ids: " + (", ".join(str(i) for i in sorted(found))
                                          or "none")), end="")
            if a.show:
                cv2.imshow("aruco", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n%s camera released" % OK)
    return 0


def report(img, a, show=False):
    found = detect_markers(img, a.dict)
    if not found:
        print("%s no markers found" % FAIL)
        print("       check: is the whole marker in frame, including its white")
        print("       border? is it in focus? is the dictionary right (%s)?" % a.dict)
        return 1
    print("%s %d marker(s)" % (OK, len(found)))
    for mid in sorted(found):
        c = found[mid]
        cx, cy = c.mean(axis=0)
        print("   id %-3d  centre (%7.1f, %7.1f)  side %6.1f px" %
              (mid, cx, cy, marker_px(c)))
    if show:
        vis = img.copy()
        for mid, c in found.items():
            cv2.polylines(vis, [c.astype(int)], True, (65, 164, 217), 2)
        cv2.imshow("aruco", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


# ------------------------------------------------------------------------ scale
def cmd_scale(a):
    img = cv2.imread(a.image)
    if img is None:
        print("%s cannot read %s" % (FAIL, a.image))
        return 1
    found = detect_markers(img, a.dict)
    if not found:
        print("%s no markers found - cannot calibrate" % FAIL)
        return 1

    sides = {mid: marker_px(c) for mid, c in found.items()}
    for mid in sorted(sides):
        print("   id %-3d  %6.1f px  ->  %6.3f px/mm" %
              (mid, sides[mid], sides[mid] / a.mm))

    vals = np.array(list(sides.values())) / a.mm
    px_mm = float(np.median(vals))
    print()
    print("%s %.4f px/mm   (1 px = %.4f mm)" % (OK, px_mm, 1.0 / px_mm))

    if len(vals) > 1:
        spread = (vals.max() - vals.min()) / px_mm * 100.0
        line = "   spread across markers: %.1f%%" % spread
        # Markers at different distances from the camera genuinely have
        # different scales. That is perspective, not noise, and no single
        # number can describe the whole frame.
        if spread > 5:
            print(line + "  " + WARN)
            print("       More than a few percent means the markers are not all")
            print("       the same distance from the camera, or the plane is")
            print("       tilted. One scale will not be right everywhere in this")
            print("       frame - rectify first, or measure near one marker.")
        else:
            print(line)

    h, w = img.shape[:2]
    c0 = found[sorted(found)[0]]
    p1, p2 = c0[0], c0[1]
    print()
    print("Paste this into your ROI Studio camera block:")
    print()
    print('  "scale": {')
    print('    "ref_points": [[%.6f, %.6f], [%.6f, %.6f]],'
          % (p1[0] / w, p1[1] / h, p2[0] / w, p2[1] / h))
    print('    "ref_length": %g,' % a.mm)
    print('    "unit": "mm"')
    print('  }')
    print()
    print("Those two points are one edge of marker id %d." % sorted(found)[0])
    print("If the camera moves afterwards, this is wrong and must be redone.")
    return 0


# ---------------------------------------------------------------------- rectify
def cmd_rectify(a):
    img = cv2.imread(a.image)
    if img is None:
        print("%s cannot read %s" % (FAIL, a.image))
        return 1
    found = detect_markers(img, a.dict)
    missing = [i for i in a.ids if i not in found]
    if missing:
        print("%s missing marker id(s): %s" % (FAIL, ", ".join(map(str, missing))))
        print("       found: %s" % (", ".join(map(str, sorted(found))) or "none"))
        return 1

    # Outer corner of each marker: top-left marker contributes its top-left
    # corner, and so on round. So the area being flattened is the span between
    # those four corners - NOT the sheet the markers are printed on.
    tl, tr, br, bl = a.ids
    src = np.array([found[tl][0], found[tr][1], found[br][2], found[bl][3]],
                   dtype=np.float32)

    out_w_mm, out_h_mm = a.size
    ppm = a.ppm
    # Keep a margin around the marker span. Two reasons: whatever you are
    # measuring usually runs right up to the markers, and a marker flush against
    # the image edge loses the white quiet zone it needs to be decoded - which
    # would also stop the self-check below from working.
    margin_mm = a.mm * 0.6
    m = int(round(margin_mm * ppm))
    W = int(round(out_w_mm * ppm)) + 2 * m
    H = int(round(out_h_mm * ppm)) + 2 * m
    dst = np.array([[m, m], [W - 1 - m, m], [W - 1 - m, H - 1 - m], [m, H - 1 - m]],
                   dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    flat = cv2.warpPerspective(img, M, (W, H))
    cv2.imwrite(a.out, flat)
    print("%s wrote %s  (%d x %d px, %g x %g mm, %g px/mm)"
          % (OK, a.out, W, H, out_w_mm, out_h_mm, ppm))

    # Check our own work: find the markers again in the flattened image and
    # confirm they come out the size they were printed. If --size was given as
    # the paper size rather than the corner-to-corner span, everything in this
    # image is stretched by that ratio and every measurement taken from it will
    # be wrong by the same amount - with nothing on screen to show it.
    back = detect_markers(flat, a.dict)
    if back:
        got = float(np.median([marker_px(c) for c in back.values()])) / ppm
        err = (got - a.mm) / a.mm * 100.0
        print()
        if abs(err) <= 2:
            print("%s markers measure %.2f mm in the result, printed at %g mm"
                  % (OK, got, a.mm))
        else:
            print("%s markers measure %.2f mm in the result, but you printed "
                  "them at %g mm" % (FAIL, got, a.mm))
            print("       Everything in this image is %.1f%% out." % err)
            print("       --size is the distance between the OUTER CORNERS of")
            print("       the four markers, not the size of the paper. Measure")
            print("       corner to corner and pass that. Try --size %.0f %.0f"
                  % (out_w_mm * a.mm / got, out_h_mm * a.mm / got))
            return 1
    else:
        print()
        print("%s could not re-find the markers to check the result - the scale"
              % WARN)
        print("       of this image has not been verified.")

    print()
    print("In this image one pixel is %g mm everywhere, which is the point:" % (1.0 / ppm))
    print("perspective is gone, so a measurement near the edge means the same")
    print("as one in the middle. Lens distortion is NOT removed - for that you")
    print("need a full camera calibration, which is a different job.")
    return 0


# ------------------------------------------------------------------------- self
def cmd_selftest(_a):
    """Render a marker, detect it back, and check the scale comes out right.

    Runs with no camera and no printer, so it is the first thing to try when
    something is not working: if this fails, the install is the problem.
    """
    fails = []
    d = get_dict(DEFAULT_DICT)
    side_px, mm = 320, 40.0
    quiet = 40
    m = cv2.aruco.generateImageMarker(d, 7, side_px)
    canvas = np.full((side_px + quiet * 2, side_px + quiet * 2), 255, np.uint8)
    canvas[quiet:quiet + side_px, quiet:quiet + side_px] = m
    img = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    found = detect_markers(img)
    print("%-44s %s" % ("marker 7 is generated and detected back",
                        OK if list(found) == [7] else FAIL))
    if list(found) != [7]:
        fails.append("detect")
    else:
        got = marker_px(found[7])
        err = abs(got - side_px) / side_px * 100
        print("%-44s %s  (%.1f px, %.2f%% off)"
              % ("its side measures what we drew", OK if err < 2 else FAIL, got, err))
        if err >= 2:
            fails.append("side")
        ppm = got / mm
        print("%-44s %s  (%.3f px/mm)"
              % ("px/mm follows from the printed size", OK if ppm > 0 else FAIL, ppm))

    rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    print("%-44s %s" % ("still found when rotated 90 degrees",
                        OK if list(detect_markers(rot)) == [7] else FAIL))
    if list(detect_markers(rot)) != [7]:
        fails.append("rotate")

    blank = np.full((300, 300, 3), 255, np.uint8)
    print("%-44s %s" % ("a blank frame reports nothing",
                        OK if not detect_markers(blank) else FAIL))
    if detect_markers(blank):
        fails.append("blank")

    print()
    if fails:
        print("%s %d check(s) failed: %s" % (FAIL, len(fails), ", ".join(fails)))
        return 1
    print("%s all checks passed" % OK)
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="ArUco markers: give the camera a known size to measure against.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dict", default=DEFAULT_DICT,
                    help="ArUco dictionary (default %s)" % DEFAULT_DICT)
    sub = ap.add_subparsers(dest="cmd")

    g = sub.add_parser("generate", help="write printable markers at an exact mm size")
    g.add_argument("--ids", type=int, nargs="+", default=[0, 1, 2, 3])
    g.add_argument("--mm", type=float, default=40.0)
    g.add_argument("--out", default="markers")
    g.set_defaults(func=cmd_generate)

    d_ = sub.add_parser("detect", help="find markers and report their IDs")
    d_.add_argument("--image")
    d_.add_argument("--camera", default="0")
    d_.add_argument("--show", action="store_true")
    d_.set_defaults(func=cmd_detect)

    s = sub.add_parser("scale", help="pixels per millimetre, and the ROI Studio block")
    s.add_argument("--image", required=True)
    s.add_argument("--mm", type=float, required=True,
                   help="the size you actually PRINTED the marker, in mm")
    s.set_defaults(func=cmd_scale)

    r = sub.add_parser("rectify", help="flatten perspective using four corner markers")
    r.add_argument("--image", required=True)
    r.add_argument("--ids", type=int, nargs=4, required=True,
                   metavar=("TL", "TR", "BR", "BL"))
    r.add_argument("--mm", type=float, default=40.0)
    r.add_argument("--size", type=float, nargs=2, required=True,
                   metavar=("W_MM", "H_MM"),
                   help="distance between the OUTER CORNERS of the four markers "
                        "in mm - not the size of the paper they are printed on")
    r.add_argument("--ppm", type=float, default=4.0, help="pixels per mm in the output")
    r.add_argument("--out", default="rectified.jpg")
    r.set_defaults(func=cmd_rectify)

    t = sub.add_parser("selftest", help="check the install with no camera")
    t.set_defaults(func=cmd_selftest)

    a = ap.parse_args()
    if not getattr(a, "func", None):
        ap.print_help()
        return 1
    if not hasattr(cv2, "aruco"):
        print("%s cv2.aruco is missing - install opencv-contrib-python" % FAIL)
        return 1
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
