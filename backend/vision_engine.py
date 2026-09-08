"""
vision_engine.py — CerebroASPECTS Unified Vision & Topographic Scoring Engine
=============================================================================
Direct Python port of CerebroASPECTS v2 client-side vision pipeline:
1. Dual-Otsu Parenchymal Isolation (air/tissue, then tissue/bone)
2. Silhouette-Based Midline Registration (distortion-free interhemispheric fissure)
3. Contralateral Attenuation Differencing with Absolute Noise Floor & Hole-Fill
4. Anatomically Precise Ganglionic ASPECTS Region Construction (Deep Nuclei & Cortical Ribbons)
5. Overlap Scoring (Strict 5% ratio threshold, zero 50px artifact)
6. Synthetic Benchmark Cohort Synthesis (normal, focal, m1)
"""

import math
import numpy as np
import cv2

SIZE = 512
DELTA = 9             # Minimum contralateral attenuation drop (8-bit scale, ~2.8 HU)
MIN_LESION_PX = 240   # Rejection floor for speckle
MIN_PARENCHYMA_AREA = 2000
INVOLVEMENT_THRESHOLD = 0.05  # 5% involvement threshold for territory infarction


def to_hu(gray_val: float) -> int:
    """Convert 8-bit display grayscale [0..255] to estimated Hounsfield Units."""
    return int(round(gray_val * 0.31 + 2))


# ── Stage 1: Dual-Otsu Segmentation ─────────────────────────

def _otsu(hist: np.ndarray, total: int) -> int:
    """Standard Otsu thresholding maximizing between-class variance."""
    if total <= 0:
        return 0
    sum_total = float(np.dot(np.arange(256), hist))
    sum_b = 0.0
    w_b = 0
    best = 0.0
    thresh = 0

    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        between = float(w_b) * float(w_f) * ((m_b - m_f) ** 2)
        if between > best:
            best = between
            thresh = t
    return thresh


def segment_parenchyma(grey: np.ndarray) -> dict:
    """
    Two-stage segmentation:
    1. First split separates head from background air.
    2. Second split over foreground separates brain soft tissue from cranial bone.
    """
    h, w = grey.shape
    assert h == SIZE and w == SIZE, f"Expected {SIZE}x{SIZE}, got {w}x{h}"

    hist, _ = np.histogram(grey, bins=256, range=(0, 256))
    head_t = _otsu(hist, grey.size)

    fg_hist = hist.copy()
    fg_hist[:head_t + 1] = 0
    fg_total = int(np.sum(fg_hist))

    bone_t = _otsu(fg_hist, fg_total) if fg_total > 0 else 255
    bone_t = int(np.clip(bone_t, head_t + 20, 250))

    mask = ((grey > head_t) & (grey < bone_t)).astype(np.uint8)

    # Retain largest connected component
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
    if num_labels <= 1:
        return None

    # Exclude background label 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    largest_area = stats[largest_idx, cv2.CC_STAT_AREA]

    if largest_area < MIN_PARENCHYMA_AREA:
        return None

    clean_mask = (labels == largest_idx).astype(np.uint8)
    bx = stats[largest_idx, cv2.CC_STAT_LEFT]
    by = stats[largest_idx, cv2.CC_STAT_TOP]
    bw = stats[largest_idx, cv2.CC_STAT_WIDTH]
    bh = stats[largest_idx, cv2.CC_STAT_HEIGHT]
    cx, cy = centroids[largest_idx]

    return {
        "mask": clean_mask,
        "head_t": head_t,
        "bone_t": bone_t,
        "area": int(largest_area),
        "bbox": {"x": int(bx), "y": int(by), "w": int(bw), "h": int(bh)},
        "centroid": {"x": float(cx), "y": float(cy)},
    }


# ── Stage 2: Silhouette Midline Registration ────────────────

def find_midline(grey: np.ndarray, seg: dict) -> dict:
    """
    Minimizes reflection asymmetry on parenchymal silhouette mask,
    preventing extensive ischemic hypodensity from dragging the midline.
    """
    bbox = seg["bbox"]
    mask = seg["mask"]
    y0 = bbox["y"] + int(bbox["h"] * 0.10)
    y1 = bbox["y"] + int(bbox["h"] * 0.90)
    guess = int(round(bbox["x"] + bbox["w"] / 2.0))
    span = max(6, min(26, int(bbox["w"] * 0.10)))

    best_x = guess
    best_cost = float("inf")

    for m in range(guess - span, guess + span + 1):
        reach = min(m - bbox["x"], bbox["x"] + bbox["w"] - m) - 2
        if reach < 24:
            continue

        mismatch = 0
        n = 0
        for y in range(y0, y1, 2):
            left_coords = m - np.arange(2, reach, 2)
            right_coords = m + np.arange(2, reach, 2)
            mismatch += np.count_nonzero(mask[y, right_coords] != mask[y, left_coords])
            n += len(left_coords)

        if n == 0:
            continue
        cost = mismatch / n
        if cost < best_cost:
            best_cost = cost
            best_x = m

    # Residual intensity asymmetry at registered midline
    reach = min(best_x - bbox["x"], bbox["x"] + bbox["w"] - best_x) - 2
    asym = 0.0
    an = 0
    for y in range(y0, y1, 2):
        for d in range(4, reach, 2):
            i_x, j_x = best_x + d, best_x - d
            if mask[y, i_x] and mask[y, j_x]:
                asym += abs(float(grey[y, i_x]) - float(grey[y, j_x]))
                an += 1

    return {
        "mid_x": int(best_x),
        "asymmetry": float(asym / an) if an > 0 else 0.0,
        "shape_cost": float(best_cost) if best_cost != float("inf") else 0.0,
    }


# ── Stage 3: Contralateral Differencing & Detection ──────────

def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Closes CSF gaps punched through infarct core using boundary flood fill."""
    h, w = mask.shape
    flood = mask.copy()
    fill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, fill_mask, (0, 0), 255)
    inverted = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, inverted)


def detect_hypodensity(grey: np.ndarray, seg: dict, mid_x: int, side: str = "right") -> np.ndarray:
    """
    Computes contralateral attenuation drop mirrored across mid_x.
    Affects ONLY the specified side (right: x > mid_x; left: x < mid_x).
    Enforces noise floor and minimum lesion volume.
    """
    mask = seg["mask"]
    diff = np.zeros((SIZE, SIZE), dtype=np.float32)
    is_right = (side.lower() == "right")

    for y in range(SIZE):
        for x in range(SIZE):
            if not mask[y, x]:
                continue
            if is_right and x <= mid_x:
                continue
            if (not is_right) and x >= mid_x:
                continue

            mx = 2 * mid_x - x
            if mx < 0 or mx >= SIZE:
                continue
            if not mask[y, mx]:
                continue

            # Contralateral drop: healthy mirrored pixel minus affected pixel
            d = float(grey[y, mx]) - float(grey[y, x])
            if d > 0:
                diff[y, x] = d

    # 2D 9x9 box blur (matches boxBlur(diff, 4))
    smooth = cv2.blur(diff, (9, 9))

    candidate_mask = ((smooth >= DELTA) & (mask > 0)).astype(np.uint8)
    if np.count_nonzero(candidate_mask) < MIN_LESION_PX:
        return np.zeros((SIZE, SIZE), dtype=np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask, connectivity=4)
    if num_labels <= 1:
        return np.zeros((SIZE, SIZE), dtype=np.uint8)

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    largest_area = stats[largest_idx, cv2.CC_STAT_AREA]

    if largest_area < MIN_LESION_PX:
        return np.zeros((SIZE, SIZE), dtype=np.uint8)

    lesion_mask = (labels == largest_idx).astype(np.uint8) * 255
    lesion_mask = fill_holes(lesion_mask)
    return lesion_mask


# ── Stage 4: Anatomical Territory Geometries ─────────────────

DEEP_STRUCTURES = [
    {
        "id": "C", "name": "Caudate Head", "group": "deep",
        "landmark": "Lateral margin of the frontal horn",
        "note": "Caudate involvement reflects ischaemia of medial lenticulostriate perforators.",
        "pts": np.array([[0.10, -0.385], [0.20, -0.345], [0.265, -0.225], [0.292, -0.085],
                         [0.243, -0.030], [0.212, -0.140], [0.163, -0.262], [0.088, -0.332]])
    },
    {
        "id": "L", "name": "Lentiform Nucleus", "group": "deep",
        "landmark": "Putamen and globus pallidus, lateral to internal capsule",
        "note": "Early loss of lentiform margin indicates established cytotoxic oedema.",
        "pts": np.array([[0.482, -0.238], [0.560, -0.170], [0.618, -0.056], [0.650, 0.058],
                         [0.594, 0.238], [0.500, 0.202], [0.456, 0.072], [0.452, -0.088]])
    },
    {
        "id": "IC", "name": "Internal Capsule", "group": "deep",
        "landmark": "Anterior limb, genu and posterior limb",
        "note": "Posterior limb carries high motor morbidity due to corticospinal tracts.",
        "pts": np.array([[0.418, -0.262], [0.366, -0.152], [0.306, -0.028], [0.380, 0.128],
                         [0.456, 0.282], [0.502, 0.220], [0.422, 0.086], [0.368, -0.030],
                         [0.416, -0.140], [0.470, -0.222]])
    },
    {
        "id": "I", "name": "Insular Ribbon", "group": "cortical",
        "landmark": "Insular cortex lining depth of Sylvian fissure",
        "note": "Earliest cortical territory to lose grey-white differentiation.",
        "pts": np.array([[0.700, -0.286], [0.756, -0.146], [0.778, -0.020], [0.772, 0.146],
                         [0.726, 0.302], [0.664, 0.286], [0.712, 0.140], [0.718, -0.014],
                         [0.700, -0.130], [0.646, -0.262]])
    }
]

CORTICAL_RIBBONS = [
    {
        "id": "M1", "name": "M1 – Anterior MCA Cortex", "group": "cortical",
        "landmark": "Anterior frontal operculum along convexity",
        "note": "Anterior cortical involvement suggests proximal/persistent occlusion.",
        "t0": -1.335, "t1": -0.505, "fi": 0.815, "fo": 0.995
    },
    {
        "id": "M2", "name": "M2 – Lateral MCA Cortex", "group": "cortical",
        "landmark": "Cortex lateral to the insular ribbon",
        "note": "Drives the greatest single reduction in salvageable tissue volume.",
        "t0": -0.425, "t1": 0.375, "fi": 0.812, "fo": 0.995
    },
    {
        "id": "M3", "name": "M3 – Posterior MCA Cortex", "group": "cortical",
        "landmark": "Posterior temporal to parieto-occipital boundary",
        "note": "Posterior cortical ischaemia approaches MCA/PCA watershed.",
        "t0": 0.455, "t1": 1.265, "fi": 0.818, "fo": 0.995
    }
]

ORDER = ["C", "L", "IC", "I", "M1", "M2", "M3"]


def _smooth_closed_points(pts: np.ndarray, iterations: int = 3) -> np.ndarray:
    """Chaikin curve corner smoothing on closed polygon."""
    cur = pts.copy()
    for _ in range(iterations):
        out = []
        n = len(cur)
        for i in range(n):
            a, b = cur[i], cur[(i + 1) % n]
            out.append(a * 0.75 + b * 0.25)
            out.append(a * 0.25 + b * 0.75)
        cur = np.array(out)
    return cur


def _build_radial_profile(seg: dict, mid_x: int, cy: float):
    """Measures parenchymal outer boundary rays and smooths profile."""
    mask = seg["mask"]
    n_rays = 360
    r_vals = np.zeros(n_rays, dtype=np.float32)

    for k in range(n_rays):
        t = (k / n_rays) * 2.0 * math.pi
        dx, dy = math.cos(t), math.sin(t)
        last_r = 0.0
        r = 4.0
        while r < 400.0:
            x = int(round(mid_x + dx * r))
            y = int(round(cy + dy * r))
            if x < 0 or y < 0 or x >= SIZE or y >= SIZE:
                break
            if mask[y, x]:
                last_r = r
            r += 1.5
        r_vals[k] = last_r

    # Circular moving average (w=7)
    s_vals = np.zeros(n_rays, dtype=np.float32)
    w = 7
    for k in range(n_rays):
        indices = [(k + d + n_rays) % n_rays for d in range(-w, w + 1)]
        s_vals[k] = np.mean(r_vals[indices])

    def radius_func(t: float) -> float:
        a = (t % (2.0 * math.pi) + 2.0 * math.pi) % (2.0 * math.pi)
        f = a / (2.0 * math.pi) * n_rays
        i = int(math.floor(f))
        frac = f - i
        return float(s_vals[i % n_rays] * (1.0 - frac) + s_vals[(i + 1) % n_rays] * frac)

    return radius_func


def build_region_masks(seg: dict, mid_x: int, side: str = "right") -> dict:
    """Constructs pixel binary masks for all 7 ASPECTS regions on affected hemisphere."""
    bbox = seg["bbox"]
    cy = seg["centroid"]["y"]
    half_w = bbox["w"] / 2.0
    half_h = bbox["h"] / 2.0
    sgn = 1.0 if side.lower() == "right" else -1.0

    masks = {}

    # Deep regions
    for r in DEEP_STRUCTURES:
        pts = r["pts"].copy()
        px_coords = np.zeros_like(pts, dtype=np.float32)
        px_coords[:, 0] = mid_x + sgn * pts[:, 0] * half_w
        px_coords[:, 1] = cy + pts[:, 1] * half_h
        smoothed = _smooth_closed_points(px_coords, 3).astype(np.int32)

        reg_mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
        cv2.fillPoly(reg_mask, [smoothed], 255)
        # Constrain to parenchyma
        reg_mask = cv2.bitwise_and(reg_mask, seg["mask"] * 255)
        masks[r["id"]] = {"mask": reg_mask, "poly": smoothed, "meta": r}

    # Cortical ribbons
    radius_at = _build_radial_profile(seg, mid_x, cy)
    n_pts = 72

    def map_t(t):
        return t if sgn > 0 else (math.pi - t)

    def at(t, f):
        R = radius_at(t)
        return [mid_x + f * R * math.cos(t), cy + f * R * math.sin(t)]

    for r in CORTICAL_RIBBONS:
        poly = []
        t0, t1 = r["t0"], r["t1"]
        # Outer arc
        for i in range(n_pts + 1):
            u = i / n_pts
            t = map_t(t0 * (1.0 - u) + t1 * u)
            poly.append(at(t, r["fo"]))
        # Inner arc
        for i in range(n_pts, -1, -1):
            u = i / n_pts
            t = map_t(t0 * (1.0 - u) + t1 * u)
            poly.append(at(t, r["fi"]))

        poly_arr = np.array(poly, dtype=np.int32)
        reg_mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
        cv2.fillPoly(reg_mask, [poly_arr], 255)
        reg_mask = cv2.bitwise_and(reg_mask, seg["mask"] * 255)
        masks[r["id"]] = {"mask": reg_mask, "poly": poly_arr, "meta": r}

    return masks


# ── Stage 5: Territory Overlap & Scoring ─────────────────────

def score_territories(grey: np.ndarray, seg: dict, region_masks: dict, lesion_mask: np.ndarray) -> dict:
    """
    Evaluates each territory with the pure 5% involvement threshold.
    Returns per-region telemetry, HU estimates, and base score.
    """
    parenchyma_mask = seg["mask"]
    region_results = {}
    base_score = 10

    for reg_id in ORDER:
        data = region_masks[reg_id]
        rmask = data["mask"]
        area = int(np.count_nonzero(rmask))

        if area > 0:
            overlap = int(np.count_nonzero((rmask > 0) & (lesion_mask > 0)))
            ratio = overlap / float(area)
        else:
            overlap = 0
            ratio = 0.0

        affected = ratio > INVOLVEMENT_THRESHOLD
        if affected:
            base_score -= 1

        # HU estimates
        lesion_in_reg = (rmask > 0) & (lesion_mask > 0) & (parenchyma_mask > 0)
        norm_in_reg = (rmask > 0) & (lesion_mask == 0) & (parenchyma_mask > 0)

        n_les = int(np.count_nonzero(lesion_in_reg))
        n_norm = int(np.count_nonzero(norm_in_reg))

        hu_les = to_hu(float(np.mean(grey[lesion_in_reg]))) if n_les > 0 else None
        hu_norm = to_hu(float(np.mean(grey[norm_in_reg]))) if n_norm > 0 else None

        region_results[reg_id] = {
            "name": data["meta"]["name"],
            "group": data["meta"]["group"],
            "landmark": data["meta"]["landmark"],
            "area": area,
            "overlap": overlap,
            "ratio": round(ratio, 4),
            "percent": round(ratio * 100.0, 1),
            "affected": bool(affected),
            "status": "INFARCTED (-1)" if affected else "INTACT",
            "hu_lesion": hu_les,
            "hu_normal": hu_norm,
        }

    return {
        "aspects_score": max(0, base_score),
        "region_results": region_results,
    }


def render_overlay(grey: np.ndarray, region_masks: dict, lesion_mask: np.ndarray) -> np.ndarray:
    """Renders clinical diagnostic overlay matching frontend styling."""
    overlay = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    tint_layer = overlay.copy()

    for reg_id, data in region_masks.items():
        poly = data["poly"]
        rmask = data["mask"]
        area = np.count_nonzero(rmask)
        overlap = np.count_nonzero((rmask > 0) & (lesion_mask > 0))
        affected = (overlap / area) > INVOLVEMENT_THRESHOLD if area > 0 else False

        color = (50, 50, 239) if affected else (80, 185, 16)  # Red vs Emerald
        cv2.fillPoly(tint_layer, [poly], color)
        cv2.polylines(overlay, [poly], True, (255, 255, 255), 1)

    # Blend regions
    blended = cv2.addWeighted(overlay, 0.70, tint_layer, 0.30, 0)

    # Highlight lesion contour in bright yellow
    if np.count_nonzero(lesion_mask) > 0:
        contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, (0, 240, 255), 2)

    return blended


# ── Full Analysis Pipeline ──────────────────────────────────

def analyze_scan_image(scan_gray: np.ndarray, hemisphere: str = "right", external_lesion_mask: np.ndarray = None) -> dict:
    """
    Complete analysis pipeline:
    - Segments parenchyma
    - Detects silhouette midline
    - Computes lesion mask (via external model if provided, or contralateral differencing)
    - Scores 7 territories
    - Generates clinical context and visual overlay
    """
    if scan_gray.shape != (SIZE, SIZE):
        scan_gray = cv2.resize(scan_gray, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)

    seg = segment_parenchyma(scan_gray)
    if seg is None:
        raise ValueError("Failed to segment brain parenchyma. Ensure CT scan has valid intracranial content.")

    midline_info = find_midline(scan_gray, seg)
    mid_x = midline_info["mid_x"]

    # Generate lesion mask
    if external_lesion_mask is not None:
        lesion_mask = external_lesion_mask
        if lesion_mask.shape != (SIZE, SIZE):
            lesion_mask = cv2.resize(lesion_mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        engine_used = "unet"
    else:
        lesion_mask = detect_hypodensity(scan_gray, seg, mid_x, side=hemisphere)
        engine_used = "contralateral_symmetry"

    region_masks = build_region_masks(seg, mid_x, side=hemisphere)
    scoring = score_territories(scan_gray, seg, region_masks, lesion_mask)
    overlay = render_overlay(scan_gray, region_masks, lesion_mask)

    # Clinical severity framing
    score = scoring["aspects_score"]
    if score == 10:
        severity = "normal"
        badge = "No Early Ischemic Changes"
        pattern_note = "No early ischemic changes identified across ganglionic MCA territories. Normal baseline."
    elif score >= 6:
        severity = "favorable"
        badge = "Favorable Ischemic Profile"
        pattern_note = "Favorable ischemic profile. Context: DAWN / DEFUSE-3 trial eligibility window; requires expert radiologist verification."
    else:
        severity = "extensive"
        badge = "Extensive Early Ischemic Changes"
        pattern_note = "Extensive early ischemic changes. Elevated reperfusion hemorrhage risk."

    return {
        "aspects_score": score,
        "max_score": 10,
        "hemisphere": hemisphere,
        "midline_x": mid_x,
        "region_results": scoring["region_results"],
        "clinical_context": {
            "severity": severity,
            "badge": badge,
            "pattern_note": pattern_note,
            "affected_territories": [k for k, v in scoring["region_results"].items() if v["affected"]],
            "intact_territories": [k for k, v in scoring["region_results"].items() if not v["affected"]],
            "disclaimer": "Observational decision-support only. Final interpretation requires radiologist verification."
        },
        "visual_overlay": overlay,
        "lesion_mask": lesion_mask,
        "telemetry": {
            "midline_x": mid_x,
            "asymmetry_signal": round(midline_info["asymmetry"], 2),
            "parenchyma_area": seg["area"],
            "lesion_pixel_count": int(np.count_nonzero(lesion_mask)),
            "engine": engine_used,
        }
    }


# ── Synthetic Benchmark Cases Generator ─────────────────────

def synthesize_benchmark_case(case_id: str, side: str = "right", offset_x: int = 0) -> np.ndarray:
    """
    Synthesizes standard benchmark CT slices identical to frontend renderCase:
    - 'normal': Healthy non-ischemic scan (score 10/10)
    - 'focal': Deep basal ganglia infarct (lentiform/capsule)
    - 'm1': Acute M1 territory infarct (dense MCA distribution)
    Supports horizontal offset for head-position invariance testing.
    """
    img = np.zeros((SIZE, SIZE), dtype=np.uint8)
    cx = 256 + offset_x
    cy = 256

    # Skull & Calvarium
    cv2.ellipse(img, (cx, cy), (185, 225), 0, 0, 360, 230, -1)  # Bone outer
    cv2.ellipse(img, (cx, cy), (175, 215), 0, 0, 360, 160, -1)  # Diploic
    cv2.ellipse(img, (cx, cy), (170, 210), 0, 0, 360, 220, -1)  # Bone inner
    cv2.ellipse(img, (cx, cy), (165, 205), 0, 0, 360, 10, -1)   # Subarachnoid CSF

    # Brain parenchyma (bilateral symmetrical baseline ~115)
    cv2.ellipse(img, (cx, cy), (160, 200), 0, 0, 360, 115, -1)

    # Ventricles (CSF ~14)
    cv2.ellipse(img, (cx - 15, cy - 20), (8, 30), 10, 0, 360, 14, -1)
    cv2.ellipse(img, (cx + 15, cy - 20), (8, 30), -10, 0, 360, 14, -1)

    # Inject pathology
    sgn = 1 if side.lower() == "right" else -1

    if case_id == "focal":
        # Lentiform / internal capsule focal hypodensity (~70 gray)
        lx = cx + sgn * 65
        ly = cy - 5
        cv2.ellipse(img, (lx, ly), (22, 18), 25 * sgn, 0, 360, 72, -1)
        img = cv2.GaussianBlur(img, (5, 5), 0)

    elif case_id == "m1":
        # Extensive MCA cortex and deep core hypodensity
        lx = cx + sgn * 90
        ly = cy - 10
        cv2.ellipse(img, (lx, ly), (55, 60), 0, 0, 360, 70, -1)
        cv2.ellipse(img, (cx + sgn * 50, cy - 10), (30, 25), 0, 0, 360, 72, -1)
        img = cv2.GaussianBlur(img, (5, 5), 0)

    return img
