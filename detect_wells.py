"""
detect_wells.py — find the circular well openings in a photo of a well plate.

Two modes:

  GRID MODE (recommended; use when you know the plate layout)
    python detect_wells.py plate.jpg --rows 8 --cols 6
    1. Build a "circular edge" response map (gradient magnitude convolved
       with a ring kernel) on a ~1000 px copy of the image.
    2. Search over well radius, pitch, rotation and orientation (rows x cols
       or cols x rows) for the lattice that best lines up with that map.
    3. Refine every well at ~2000 px using ONE shared radius (all openings
       are the same size), then fit a perspective homography through the
       refined centres with RANSAC. Outliers (glare, the bottom rim of a
       clear well, reflections) are rejected, and every well's final position
       comes from the fitted grid, so they're all geometrically consistent.
    4. Find the top OPENINGS: larger circles, shifted from the floors by
       perspective when the camera isn't exactly overhead. The shift is
       measured per well, then smoothed with a robust linear model.

  FREE MODE (no layout given): plain Hough circles. Fine for opaque plates
  with strong contrast; unreliable for clear plastic plates, where each well
  shows both its top and bottom rim.

Outputs: JSON/CSV of wells: label, OPENING centre/radius (x, y, r), well-floor
centre/radius (floor_x, floor_y, floor_r) and a 'fit_ok' flag,
and an annotated image. Coordinates are in ORIGINAL image pixels.

Input: an image file (.jpg/.png, or iPhone .HEIC via heic_reader.py), or a
live frame from a camera:
    python detect_wells.py --camera 0 --rows 8 --cols 6
Or call it from your own robot code:
    from detect_wells import capture_frame, find_wells
    frame = capture_frame(0)
    wells = find_wells(frame, rows=8, cols=6)
"""

import argparse
import json
import os
import string
import sys

import cv2
import numpy as np


# ---------------------------------------------------------------- I/O
def load_image(path):
    img = cv2.imread(path)
    if img is None and path.lower().endswith((".heic", ".heif")):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from heic_reader import read_heic
        img = read_heic(path)
    if img is None:
        sys.exit(f"Could not read {path}")
    return img


def resize_max(img, max_dim):
    s = max_dim / max(img.shape[:2])
    s = min(s, 1.0)
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), s


# ---------------------------------------------------------------- preprocessing
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.medianBlur(gray, 5)


def edge_magnitude(img):
    g = preprocess(img).astype(np.float32)
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def ring_kernel(r, thickness=2.0, contrast=True, inner_only=False):
    """Ring of radius r. With contrast=True it is zero-mean: the rim must be
    stronger than the bands just inside and outside it, so busy texture
    (reflections, scratches, the plate frame) scores ~0 instead of high."""
    k = int(r * 1.35 + 3 * thickness + 1)
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    d = np.hypot(xx, yy)

    def band(rad):
        B = np.exp(-(((d - rad) / thickness) ** 2) / 2)
        return B / B.sum()

    K = band(r)
    if contrast and inner_only:
        K = K - band(r * 0.8)
    elif contrast:
        K = K - 0.5 * band(r * 0.7) - 0.5 * band(r * 1.3)
    return K.astype(np.float32)


# ---------------------------------------------------------------- grid mode
def _lattice_points(rows, cols, pitch, angle_deg):
    a = np.deg2rad(angle_deg)
    ux = np.array([np.cos(a), np.sin(a)]) * pitch     # step along a row
    uy = np.array([-np.sin(a), np.cos(a)]) * pitch    # step down a column
    pts = np.array([i * uy + j * ux for i in range(rows) for j in range(cols)])
    return pts - pts.min(0)


def _lattice_score(resp, rows, cols, pitch, angle):
    pts = _lattice_points(rows, cols, pitch, angle)
    w, h = (np.ceil(pts.max(0)) + 2).astype(int)
    if h >= resp.shape[0] or w >= resp.shape[1]:
        return -1.0, None
    K = np.zeros((h, w), np.float32)
    for x, y in pts:
        K[int(round(y)), int(round(x))] = 1.0
    S = cv2.matchTemplate(resp, K, cv2.TM_CCORR)
    _, m, _, loc = cv2.minMaxLoc(S)
    return m / len(pts), pts + np.array(loc)


def _search(mag, rows, cols, radii, pitch_ratios, angles, orientations):
    best = None
    for r in radii:
        resp = cv2.filter2D(mag, -1, ring_kernel(r))
        for (nr, nc) in orientations:
            for k in pitch_ratios:
                for a in angles:
                    sc, pts = _lattice_score(resp, nr, nc, r * k, a)
                    if pts is not None and (best is None or sc > best["score"]):
                        best = dict(score=sc, r=r, rows=nr, cols=nc,
                                    pitch=r * k, angle=a, pts=pts)
    return best


def fit_grid(img, rows, cols, min_r=None, max_r=None, allow_transpose=True,
             verbose=True):
    # ---- stage 1: coarse lattice search at ~1000 px
    small, s1 = resize_max(img, 1000)
    mag = edge_magnitude(small)
    h, w = mag.shape
    lo = (min_r * s1) if min_r else min(h, w) / 100
    hi = (max_r * s1) if max_r else min(h, w) / (2.05 * max(rows, cols))
    orients = [(rows, cols), (cols, rows)] if (allow_transpose and rows != cols) else [(rows, cols)]
    best = _search(mag, rows, cols, np.geomspace(lo, hi, 14),
                   np.linspace(2.05, 2.8, 10), np.arange(-4, 4.1, 2), orients)
    if best is None:
        raise RuntimeError("Grid search failed; check --rows/--cols or --min-r/--max-r")
    # fine search around the best coarse hit
    best = _search(mag, rows, cols,
                   best["r"] * np.linspace(0.93, 1.07, 5),
                   (best["pitch"] / best["r"]) * np.linspace(0.96, 1.04, 5),
                   best["angle"] + np.arange(-1.5, 1.6, 0.5),
                   [(best["rows"], best["cols"])])
    rows, cols = best["rows"], best["cols"]
    if verbose:
        print(f"Lattice: {rows}x{cols}, radius~{best['r']/s1:.0f}px, "
              f"pitch~{best['pitch']/s1:.0f}px, angle {best['angle']:+.1f} deg")

    # ---- stage 2: per-well refinement at ~2000 px with one shared radius
    big, s2 = resize_max(img, 2000)
    M = edge_magnitude(big)
    k = s2 / s1
    P = best["pts"] * k
    win = int(best["pitch"] * k * 0.15)
    pad = int(best["r"] * k * 1.6) + 10
    Mp = cv2.copyMakeBorder(M, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    def refine(rr):
        K = ring_kernel(rr, 2.5)
        hk = K.shape[0] // 2
        out, tot = [], 0.0
        for x, y in P:
            x0, y0 = int(x) - win + pad, int(y) - win + pad
            sub = Mp[y0 - hk:y0 + 2 * win + hk + 1, x0 - hk:x0 + 2 * win + hk + 1]
            S = cv2.matchTemplate(sub, K, cv2.TM_CCORR)
            _, m, _, l = cv2.minMaxLoc(S)
            out.append((x0 - pad + l[0], y0 - pad + l[1]))
            tot += m
        return tot, np.array(out, np.float32)

    r2 = best["r"] * k
    trials = [(refine(rr), rr) for rr in np.linspace(r2 * 0.8, r2 * 1.4, 16)]
    (_, Q), R = max(trials, key=lambda t: t[0][0])

    # ---- stage 3: perspective grid through refined centres (well FLOORS)
    # The strongest circular edge in each well is usually the floor / inner
    # wall, so this grid is the most reliable anchor.
    ideal = np.array([(j, i) for i in range(rows) for j in range(cols)], np.float32)
    H, mask = cv2.findHomography(ideal, Q, cv2.RANSAC, best["pitch"] * k * 0.06)
    Ffloor = cv2.perspectiveTransform(ideal[None], H)[0]
    if verbose:
        print(f"Floor grid: {int(mask.sum())}/{len(mask)} wells agree")

    # ---- stage 4: top OPENINGS
    # Openings are bigger (radius ~0.45-0.5 x pitch on most plates) and, unless
    # the camera is exactly overhead, shifted from the floor by perspective.
    # The shift varies smoothly across the plate, so: measure it per well, then
    # fit a linear offset field with robust (outlier-downweighting) least squares.
    pitch2 = best["pitch"] * k
    Ropen, Fopen, open_ok = _fit_openings(M, Ffloor, pitch2, R)
    if verbose:
        print(f"Openings: radius~{Ropen/s2:.0f}px, max perspective shift "
              f"{np.hypot(*(Fopen-Ffloor).T).max()/s2:.0f}px, "
              f"{int(open_ok.sum())}/{len(open_ok)} wells agree")

    wells = []
    for n in range(len(ideal)):
        i, j = divmod(n, cols)
        (x, y), (fx, fy) = Fopen[n] / s2, Ffloor[n] / s2
        wells.append(dict(row=i, col=j,
                          x=round(float(x), 1), y=round(float(y), 1), r=round(float(Ropen / s2), 1),
                          floor_x=round(float(fx), 1), floor_y=round(float(fy), 1),
                          floor_r=round(float(R / s2), 1),
                          fit_ok=bool(open_ok[n] and mask.ravel()[n])))
    return wells, rows, cols


def _fit_openings(M, P, pitch, floor_r, iters=3):
    win = int(pitch * 0.3)
    pad = int(pitch * 0.9) + 20
    Mp = cv2.copyMakeBorder(M, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    A = np.c_[np.ones(len(P)), (P - P.mean(0)) / pitch]

    def measure(rr):
        K = ring_kernel(rr, 2.0, inner_only=True)
        hk = K.shape[0] // 2
        out = []
        for x, y in P:
            x0, y0 = int(x) - win + pad, int(y) - win + pad
            S = cv2.matchTemplate(Mp[y0 - hk:y0 + 2 * win + hk + 1, x0 - hk:x0 + 2 * win + hk + 1],
                                  K, cv2.TM_CCORR)
            _, _, _, l = cv2.minMaxLoc(S)
            out.append((x0 - pad + l[0], y0 - pad + l[1]))
        return np.array(out, np.float32)

    def smooth(Q):
        D = Q - P
        w = np.ones(len(P))
        for _ in range(10):   # iteratively reweighted least squares
            coef = np.linalg.lstsq(A * w[:, None], D * w[:, None], rcond=None)[0]
            res = np.hypot(*(D - A @ coef).T)
            sc = max(np.median(res) * 1.5, 1.0)
            w = np.minimum(1, sc / np.maximum(res, 1e-6))
        return P + A @ coef, res

    def best_radius(C):
        # the opening must be larger than the floor rim, and can't exceed half the pitch
        rs = np.arange(max(pitch * 0.35, floor_r * 1.05), pitch * 0.56, 0.5)
        th = np.linspace(0, 2 * np.pi, 240, endpoint=False)
        h, w = M.shape
        prof = []
        for r in rs:
            xs = np.clip((C[:, :1] + r * np.cos(th)).astype(int), 0, w - 1)
            ys = np.clip((C[:, 1:] + r * np.sin(th)).astype(int), 0, h - 1)
            prof.append(M[ys, xs].mean())
        return rs[int(np.argmax(prof))]

    R = max(pitch * 0.48, floor_r * 1.1)
    F = P.copy()
    for _ in range(iters):
        F, res = smooth(measure(R))
        R = best_radius(F)
    F, res = smooth(measure(R))
    return R, F, res < pitch * 0.12


# ---------------------------------------------------------------- camera + API
def capture_frame(camera=0, width=None, height=None, warmup=15):
    """Grab one frame from a USB/built-in camera (OpenCV VideoCapture).
    camera: device index (0 = first camera) or a stream URL.
    warmup: frames to discard first so auto-exposure/focus can settle."""
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera!r}")
    try:
        if width and height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        frame = None
        for _ in range(max(1, warmup)):
            ok, f = cap.read()
            if ok:
                frame = f
        if frame is None:
            raise RuntimeError(f"Camera {camera!r} opened but returned no frames")
        return frame
    finally:
        cap.release()


def find_wells(img, rows=None, cols=None, min_r=None, max_r=None, param2=30,
               allow_transpose=True, letters_on="rows", flip_rows=False,
               flip_cols=False, verbose=False):
    """Detect wells in a BGR image (numpy array). Returns a list of dicts."""
    if rows and cols:
        wells, r, c = fit_grid(img, rows, cols, min_r, max_r,
                               allow_transpose=allow_transpose, verbose=verbose)
        return apply_labels(wells, r, c, letters_on, flip_rows, flip_cols)
    return detect_circles_free(img, min_r, max_r, param2)


# ---------------------------------------------------------------- free mode
def detect_circles_free(img, min_r=None, max_r=None, param2=30):
    small, s = resize_max(img, 1200)
    gray = preprocess(small)
    h, w = gray.shape
    lo = int(min_r * s) if min_r else max(5, int(min(h, w) / 60))
    hi = int(max_r * s) if max_r else int(min(h, w) / 10)
    c = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=int(lo * 1.8),
                         param1=100, param2=param2, minRadius=lo, maxRadius=hi)
    if c is None:
        return []
    c = c[0]
    med = np.median(c[:, 2])
    c = c[np.abs(c[:, 2] - med) <= 0.25 * med] / s
    order = np.lexsort((c[:, 0], np.round(c[:, 1] / np.median(c[:, 2]))))
    return [dict(id=n, x=round(float(x), 1), y=round(float(y), 1), r=round(float(r), 1))
            for n, (x, y, r) in enumerate(c[order])]


# ---------------------------------------------------------------- labelling
def letter(i):
    L = string.ascii_uppercase
    return L[i] if i < 26 else L[i // 26 - 1] + L[i % 26]


def apply_labels(wells, rows, cols, letters_on="rows", flip_rows=False, flip_cols=False):
    """Plate labels (A1 etc.) in the image's orientation.
    letters_on='rows': A,B,C.. go down image rows, numbers across columns.
    letters_on='cols': letters go across image columns (plate photographed rotated)."""
    for w in wells:
        i = rows - 1 - w["row"] if flip_rows else w["row"]
        j = cols - 1 - w["col"] if flip_cols else w["col"]
        w["label"] = f"{letter(i)}{j+1}" if letters_on == "rows" else f"{letter(j)}{i+1}"
    return wells


# ---------------------------------------------------------------- output
def annotate(img, wells):
    out = img.copy()
    t = max(2, int(max(img.shape[:2]) / 800))
    fs = max(0.5, max(img.shape[:2]) / 2500)
    for w in wells:
        c = (int(w["x"]), int(w["y"]))
        color = (0, 200, 0) if w.get("fit_ok", True) else (0, 165, 255)
        if "floor_x" in w:
            cv2.circle(out, (int(w["floor_x"]), int(w["floor_y"])), int(w["floor_r"]),
                       (200, 200, 200), max(1, t // 2))
        cv2.circle(out, c, int(w["r"]), color, t)
        cv2.circle(out, c, t * 2, (0, 0, 255), -1)
        if "label" in w:
            cv2.putText(out, w["label"], (c[0] - int(w["r"] * 0.4), c[1] - int(w["r"]) - t * 3),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 0, 0), t)
    return out


def main():
    ap = argparse.ArgumentParser(description="Detect circular wells in a tray image.")
    ap.add_argument("image", nargs="?", help="image file (omit when using --camera)")
    ap.add_argument("--camera", help="camera index (e.g. 0) or stream URL to capture from")
    ap.add_argument("--resolution", help="camera resolution as WIDTHxHEIGHT, e.g. 1920x1080")
    ap.add_argument("--save-frame", help="also save the captured camera frame to this file")
    ap.add_argument("--rows", type=int, help="well rows as seen in the image")
    ap.add_argument("--cols", type=int, help="well columns as seen in the image")
    ap.add_argument("--no-transpose", action="store_true",
                    help="don't also try rows/cols swapped")
    ap.add_argument("--letters-on", choices=["rows", "cols"], default="rows",
                    help="which image axis carries the plate's letter labels")
    ap.add_argument("--flip-rows", action="store_true", help="number rows bottom-up")
    ap.add_argument("--flip-cols", action="store_true", help="number cols right-to-left")
    ap.add_argument("--min-r", type=float, help="min well radius (original px)")
    ap.add_argument("--max-r", type=float, help="max well radius (original px)")
    ap.add_argument("--param2", type=int, default=30, help="Hough sensitivity (free mode)")
    ap.add_argument("--out", default="wells.json", help=".json or .csv")
    ap.add_argument("--annotated", default="wells_annotated.jpg")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if args.camera is not None:
        cam = int(args.camera) if args.camera.isdigit() else args.camera
        wh = [int(v) for v in args.resolution.lower().split("x")] if args.resolution else (None, None)
        img = capture_frame(cam, *wh)
        if args.save_frame:
            cv2.imwrite(args.save_frame, img)
    elif args.image:
        img = load_image(args.image)
    else:
        ap.error("give an image file or --camera INDEX")
    print(f"Image {img.shape[1]}x{img.shape[0]}")

    wells = find_wells(img, args.rows, args.cols, args.min_r, args.max_r, args.param2,
                       not args.no_transpose, args.letters_on, args.flip_rows,
                       args.flip_cols, verbose=True)
    if not (args.rows and args.cols):
        print(f"Free mode: {len(wells)} circles (pass --rows/--cols for reliable results)")
    if not wells:
        sys.exit("No wells found.")

    if args.out.endswith(".csv"):
        import csv
        with open(args.out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(wells[0].keys()))
            wr.writeheader()
            wr.writerows(wells)
    else:
        with open(args.out, "w") as f:
            json.dump(wells, f, indent=2)
    print(f"Wrote {len(wells)} wells to {args.out}")

    vis = annotate(img, wells)
    cv2.imwrite(args.annotated, vis)
    print(f"Annotated image: {args.annotated}")
    if args.show:
        cv2.imshow("wells", resize_max(vis, 1200)[0])
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
