import cv2
import numpy as np
import os

def visualize(img_path, mask_path, out_path):
    img = cv2.imread(img_path)
    mask = np.load(mask_path)
    
    # Create red overlay
    overlay = img.copy()
    overlay[mask == 1] = [0, 0, 255]  # Red for forged
    
    # Blend images
    alpha = 0.5
    res = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    
    # Add a title
    cv2.putText(res, f"Forged Px: {mask.sum()}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    cv2.imwrite(out_path, res)
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python visualize.py <img_path> <mask_path> [out_path]")
        sys.exit(1)
    
    img_p = sys.argv[1]
    mask_p = sys.argv[2]
    out_p = sys.argv[3] if len(sys.argv) > 3 else "predictions/vis.png"
    
    visualize(img_p, mask_p, out_p)
