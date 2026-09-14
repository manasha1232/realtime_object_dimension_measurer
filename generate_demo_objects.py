import os
import cv2
import numpy as np

def generate_synthetic_dimension_test_image(output_path="input/sample_objects_table.jpg", width=1280, height=960):
    """
    Generates a synthetic test image with multiple objects of known physical dimensions:
    1. Reference Object: Credit Card (85.6 mm x 54.0 mm) placed at top-left
    2. Smartphone: (140 mm x 70 mm)
    3. Sticky Note: (75 mm x 75 mm)
    4. Coin / Circle Marker: (30 mm diameter)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create dark wood texture desk background
    bg = np.ones((height, width, 3), dtype=np.uint8) * 40
    
    # Wood grain lines
    for y in range(0, height, 4):
        bg[y, :, :] = np.clip(bg[y, :, :] + np.random.randint(0, 15), 0, 255)
        
    np.random.seed(42)
    noise = np.random.randint(-10, 10, (height, width, 3), dtype=np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Grid scale for reference: 1 mm ~= 3 pixels
    px_per_mm = 3.2
    
    # Object 1: Reference Credit Card (85.6 mm x 54.0 mm -> 274 px x 173 px)
    card_w_px = int(85.6 * px_per_mm)
    card_h_px = int(54.0 * px_per_mm)
    card_img = np.ones((card_h_px, card_w_px, 3), dtype=np.uint8) * 240
    cv2.rectangle(card_img, (0, 0), (card_w_px, card_h_px), (180, 50, 20), -1) # Blue card
    cv2.putText(card_img, "REF: CREDIT CARD", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(card_img, "85.6mm x 54.0mm", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    
    # Object 2: Smartphone (140 mm x 70 mm -> 448 px x 224 px)
    phone_w_px = int(70 * px_per_mm)
    phone_h_px = int(140 * px_per_mm)
    phone_img = np.ones((phone_h_px, phone_w_px, 3), dtype=np.uint8) * 20
    cv2.rectangle(phone_img, (10, 10), (phone_w_px - 10, phone_h_px - 10), (220, 220, 220), 2)
    cv2.putText(phone_img, "PHONE", (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    # Object 3: Sticky Note (75 mm x 75 mm -> 240 px x 240 px)
    note_size_px = int(75 * px_per_mm)
    note_img = np.ones((note_size_px, note_size_px, 3), dtype=np.uint8)
    note_img[:] = (50, 240, 255) # Bright Yellow note
    cv2.putText(note_img, "STICKY NOTE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2)
    cv2.putText(note_img, "75mm x 75mm", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
    
    # Helper to paste rotated object into background
    def paste_rotated(bg_img, obj, pos_x, pos_y, angle_deg):
        oh, ow = obj.shape[:2]
        M = cv2.getRotationMatrix2D((ow / 2.0, oh / 2.0), angle_deg, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((oh * sin) + (ow * cos))
        nH = int((oh * cos) + (ow * sin))
        M[0, 2] += (nW / 2) - (ow / 2)
        M[1, 2] += (nH / 2) - (oh / 2)
        rotated = cv2.warpAffine(obj, M, (nW, nH), borderValue=(0, 0, 0))
        
        mask = (rotated[:, :, 0] > 0) | (rotated[:, :, 1] > 0) | (rotated[:, :, 2] > 0)
        mask_3c = cv2.merge([mask, mask, mask])
        
        y1, y2 = pos_y, pos_y + nH
        x1, x2 = pos_x, pos_x + nW
        bg_img[y1:y2, x1:x2] = np.where(mask_3c, rotated, bg_img[y1:y2, x1:x2])
        
    # Paste Objects on Table Canvas
    paste_rotated(bg, card_img, 120, 150, 15)     # Credit Card (Tilted 15 deg)
    paste_rotated(bg, phone_img, 550, 120, -25)   # Smartphone (Tilted -25 deg)
    paste_rotated(bg, note_img, 200, 550, 10)     # Sticky Note (Tilted 10 deg)
    
    # Object 4: Circular Coin (30 mm diameter)
    coin_r = int(15 * px_per_mm)
    cv2.circle(bg, (800, 650), coin_r, (180, 180, 180), -1)
    cv2.circle(bg, (800, 650), coin_r, (240, 240, 240), 3)
    cv2.putText(bg, "COIN (30mm)", (735, 655), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 2)
    
    # Title Banner
    cv2.putText(bg, "OBJECT DIMENSION MEASUREMENT TEST BENCHMARK", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 255), 2)
                
    cv2.imwrite(output_path, bg)
    print(f"[OK] Synthetic objects image saved to '{output_path}'")
    return output_path

if __name__ == "__main__":
    generate_synthetic_dimension_test_image()
