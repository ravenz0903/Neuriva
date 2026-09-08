import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from aspects_engine import extract_brain_bbox, NORMALIZED_REGIONS

def preview_polygons(image_path="dummy_data/normal_scan.png", hemisphere="right"):
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}. Run generate_dummy_data.py first!")
        return
        
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    bx, by, bw, bh = extract_brain_bbox(img)
    mid_x = bx + bw // 2
    
    if hemisphere == "right":
        hx, hy, hw, hh = mid_x, by, bw // 2, bh
    else:
        hx, hy, hw, hh = bx, by, bw // 2, bh
        
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Draw brain bbox
    cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (255, 255, 0), 1)
    cv2.line(vis, (mid_x, by), (mid_x, by + bh), (255, 255, 0), 1)
    
    for name, p_norm in NORMALIZED_REGIONS.items():
        px = np.zeros_like(p_norm, dtype=np.int32)
        if hemisphere == "right":
            px[:, 0] = (hx + p_norm[:, 0] * hw).astype(np.int32)
        else:
            px[:, 0] = (hx + (1.0 - p_norm[:, 0]) * hw).astype(np.int32)
        px[:, 1] = (hy + p_norm[:, 1] * hh).astype(np.int32)
        
        cv2.polylines(vis, [px], True, (0, 255, 0), 2)
        cv2.putText(vis, name.split()[0], (px[0, 0], px[0, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    output_preview = "polygon_preview.png"
    cv2.imwrite(output_preview, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"Saved preview visualization to: {output_preview}")

if __name__ == "__main__":
    preview_polygons()
