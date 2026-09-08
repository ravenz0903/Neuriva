import cv2
import numpy as np
import os

def create_synthetic_slice():
    """Generates a realistic 512x512 axial brain slice representation."""
    img = np.zeros((512, 512), dtype=np.uint8)
    
    # Brain tissue (parenchyma)
    cv2.ellipse(img, (256, 256), (175, 215), 0, 0, 360, 135, -1)
    
    # Skull bone
    cv2.ellipse(img, (256, 256), (180, 220), 0, 0, 360, 220, 4)
    
    # Ventricles (CSF - darker fluid in center)
    cv2.ellipse(img, (240, 240), (12, 45), -15, 0, 360, 50, -1)
    cv2.ellipse(img, (272, 240), (12, 45), 15, 0, 360, 50, -1)
    
    # Interhemispheric fissure (midline subtle groove)
    cv2.line(img, (256, 50), (256, 460), 70, 1)
    
    return img

def main():
    output_dir = "dummy_data"
    os.makedirs(output_dir, exist_ok=True)
    
    base_scan = create_synthetic_slice()
    cv2.imwrite(os.path.join(output_dir, "normal_scan.png"), base_scan)
    
    # Case 1: Normal (empty mask)
    mask_normal = np.zeros((512, 512), dtype=np.uint8)
    cv2.imwrite(os.path.join(output_dir, "mask_normal.png"), mask_normal)
    
    # Case 2: Focal Subcortical Infarct (Right Lentiform & Internal Capsule)
    # Midline is at x=256. Right hemisphere is x > 256.
    mask_focal = np.zeros((512, 512), dtype=np.uint8)
    cv2.circle(mask_focal, (305, 250), 22, 255, -1)
    cv2.imwrite(os.path.join(output_dir, "mask_focal.png"), mask_focal)
    
    # Case 3: Large MCA Infarct (Right Cortical M1, M2 + Subcortical)
    mask_large = np.zeros((512, 512), dtype=np.uint8)
    cv2.ellipse(mask_large, (340, 260), (45, 75), 0, 0, 360, 255, -1)
    cv2.imwrite(os.path.join(output_dir, "mask_large.png"), mask_large)
    
    print(f"Generated 3 synthetic test cases in '{output_dir}/':")
    print("  1. normal_scan.png + mask_normal.png (Expected ASPECTS: 10/10)")
    print("  2. mask_focal.png (Expected ASPECTS: ~8/10)")
    print("  3. mask_large.png (Expected ASPECTS: <= 5/10)")

if __name__ == "__main__":
    main()
