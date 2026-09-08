import cv2
import numpy as np

NORMALIZED_REGIONS = {
    "Caudate (C)": np.array([[0.10, 0.40], [0.25, 0.38], [0.25, 0.50], [0.12, 0.50]]),
    "Lentiform (L)": np.array([[0.28, 0.42], [0.48, 0.40], [0.45, 0.58], [0.28, 0.55]]),
    "Internal Capsule (IC)": np.array([[0.20, 0.50], [0.28, 0.42], [0.32, 0.58], [0.22, 0.62]]),
    "Insular Ribbon (I)": np.array([[0.50, 0.38], [0.60, 0.38], [0.58, 0.62], [0.48, 0.60]]),
    "M1 (Anterior MCA)": np.array([[0.30, 0.15], [0.65, 0.18], [0.55, 0.32], [0.25, 0.28]]),
    "M2 (Lateral MCA)": np.array([[0.62, 0.35], [0.90, 0.38], [0.88, 0.65], [0.60, 0.62]]),
    "M3 (Posterior MCA)": np.array([[0.30, 0.72], [0.65, 0.68], [0.85, 0.66], [0.50, 0.88]]),
}

def extract_brain_bbox(image_gray):
    _, thresh = cv2.threshold(image_gray, 20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = image_gray.shape
        return 0, 0, w, h
    largest = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(largest)

def compute_aspects(scan_gray, lesion_mask, affected_hemisphere="right"):
    bx, by, bw, bh = extract_brain_bbox(scan_gray)
    midline_x = bx + bw // 2
    
    if affected_hemisphere == "right":
        hx, hy, hw, hh = midline_x, by, bw // 2, bh
    else:
        hx, hy, hw, hh = bx, by, bw // 2, bh

    base_score = 10
    region_results = {}
    overlay = cv2.cvtColor(scan_gray, cv2.COLOR_GRAY2BGR)
    overlay_mask = np.zeros_like(overlay)

    for name, poly_norm in NORMALIZED_REGIONS.items():
        poly_px = np.zeros_like(poly_norm, dtype=np.int32)
        if affected_hemisphere == "right":
            poly_px[:, 0] = (hx + poly_norm[:, 0] * hw).astype(np.int32)
        else:
            poly_px[:, 0] = (hx + (1.0 - poly_norm[:, 0]) * hw).astype(np.int32)
        poly_px[:, 1] = (hy + poly_norm[:, 1] * hh).astype(np.int32)

        reg_mask = np.zeros(scan_gray.shape, dtype=np.uint8)
        cv2.fillPoly(reg_mask, [poly_px], 255)

        region_area = np.count_nonzero(reg_mask)
        overlap_area = np.count_nonzero(cv2.bitwise_and(lesion_mask, reg_mask))
        overlap_ratio = (overlap_area / region_area) if region_area > 0 else 0.0

        is_affected = overlap_ratio > 0.05 or overlap_area > 50

        if is_affected:
            base_score -= 1
            color = (0, 0, 255)
            status = "INFARCTED (-1)"
        else:
            color = (0, 255, 0)
            status = "INTACT"

        region_results[name] = {
            "status": status,
            "overlap_ratio": round(overlap_ratio * 100, 2),
            "affected": is_affected
        }

        cv2.fillPoly(overlay_mask, [poly_px], color)
        cv2.polylines(overlay, [poly_px], True, (255, 255, 255), 1)

    lesion_contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, lesion_contours, -1, (0, 255, 255), 2)
    final_view = cv2.addWeighted(overlay, 0.75, overlay_mask, 0.25, 0)

    return {
        "aspects_score": max(0, base_score),
        "region_results": region_results,
        "visual_overlay": final_view
    }

if __name__ == "__main__":
    dummy_scan = np.zeros((512, 512), dtype=np.uint8)
    cv2.ellipse(dummy_scan, (256, 256), (180, 220), 0, 0, 360, 140, -1)
    cv2.ellipse(dummy_scan, (256, 256), (185, 225), 0, 0, 360, 200, 3)

    dummy_lesion = np.zeros((512, 512), dtype=np.uint8)
    cv2.circle(dummy_lesion, (310, 250), 30, 255, -1)

    res = compute_aspects(dummy_scan, dummy_lesion, affected_hemisphere="right")
    print(f"ASPECTS Score: {res['aspects_score']} / 10")
    for r, d in res["region_results"].items():
        print(f"  {r}: {d['status']} ({d['overlap_ratio']}%)")
