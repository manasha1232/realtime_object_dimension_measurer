#!/usr/bin/env python3
"""
===============================================================================
Real-Time Object Dimension Measurer
Day 12 - 30-Day Computer Vision & Deep Learning Challenge
===============================================================================
Author: Computer Vision & AI Agent
Technologies: OpenCV, Contours, Rotated Bounding Boxes (minAreaRect), Homography

Description:
    Real-time object dimension measurement system. Uses reference object scale 
    calibration, rotated minimum bounding box fitting (cv2.minAreaRect), Euclidean 
    edge distance computation, and dimension annotations (mm, cm, inches).
===============================================================================
"""

import os
import sys
import glob
import json
import time
import argparse
import cv2
import numpy as np


def order_points(pts):
    """
    Orders 4 bounding box points in sequence:
    Top-Left (TL), Top-Right (TR), Bottom-Right (BR), Bottom-Left (BL).
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def midpoint(ptA, ptB):
    """Calculates midpoint between two 2D points."""
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def measure_object_dimensions(image, ref_width_mm=85.6, min_area=1500):
    """
    Detects objects and measures real-world dimensions (width, height) using reference scale.
    
    Args:
        image (np.ndarray): Input BGR image.
        ref_width_mm (float): Known physical width of reference object in mm (default: 85.6 mm for Credit Card).
        min_area (int): Minimum contour area threshold.
        
    Returns:
        tuple: (annotated_image, edges_mask, list_of_measured_objects, pixels_per_mm)
    """
    vis = image.copy()
    h, w = vis.shape[:2]
    
    # 1. Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Canny edge detection
    edged = cv2.Canny(blurred, 30, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged = cv2.dilate(edged, kernel, iterations=1)
    edged = cv2.erode(edged, kernel, iterations=1)
    
    # 2. Contour Extraction
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours left-to-right by X coordinate
    valid_contours = []
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            box = cv2.minAreaRect(c)
            valid_contours.append((c, box))
            
    valid_contours = sorted(valid_contours, key=lambda item: item[1][0][0]) # Sort by center X
    
    pixels_per_mm = None
    measured_objects = []
    
    for idx, (c, rect) in enumerate(valid_contours):
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype="int")
        box = order_points(box)
        
        (tl, tr, br, bl) = box
        
        # Calculate midpoints
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)
        
        # Calculate Euclidean distances in pixels
        height_px = np.hypot(tltrX - blbrX, tltrY - blbrY)
        width_px  = np.hypot(tlblX - trbrX, tlblY - trbrY)
        
        # First object is calibration reference
        if pixels_per_mm is None:
            pixels_per_mm = max(width_px, height_px) / float(ref_width_mm)
            is_ref = True
        else:
            is_ref = False
            
        # Convert pixels to real-world measurements
        width_mm  = width_px / pixels_per_mm
        height_mm = height_px / pixels_per_mm
        
        width_cm  = width_mm / 10.0
        height_cm = height_mm / 10.0
        
        width_in  = width_mm / 25.4
        height_in = height_mm / 25.4
        
        obj_info = {
            "id": idx + 1,
            "is_reference": is_ref,
            "dimensions_mm": {"width": round(float(width_mm), 1), "height": round(float(height_mm), 1)},
            "dimensions_cm": {"width": round(float(width_cm), 2), "height": round(float(height_cm), 2)},
            "dimensions_in": {"width": round(float(width_in), 2), "height": round(float(height_in), 2)},
            "bounding_box": [[round(float(pt[0]), 1), round(float(pt[1]), 1)] for pt in box]
        }
        measured_objects.append(obj_info)
        
        # Draw Rotated Bounding Box
        box_color = (0, 255, 120) if is_ref else (0, 215, 255)
        cv2.drawContours(vis, [box.astype("int")], -1, box_color, 2, lineType=cv2.LINE_AA)
        
        # Draw corner dots
        for pt in box:
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
            
        # Draw midpoint leader lines
        cv2.line(vis, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 1, lineType=cv2.LINE_AA)
        cv2.line(vis, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 1, lineType=cv2.LINE_AA)
        
        # Draw Dimension Text Tag
        tag = f"REF #{idx+1}" if is_ref else f"OBJ #{idx+1}"
        dim_str = f"{width_cm:.1f}x{height_cm:.1f}cm ({width_in:.1f}x{height_in:.1f}in)"
        
        cx, cy = int(rect[0][0]), int(rect[0][1])
        cv2.putText(vis, tag, (cx - 40, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2, lineType=cv2.LINE_AA)
        cv2.putText(vis, dim_str, (cx - 70, cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        
    # Top HUD Banner
    banner_h = 50
    banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
    banner[:] = (20, 20, 20)
    
    scale_str = f"SCALE: {pixels_per_mm:.2f} px/mm" if pixels_per_mm else "SCALE: UNCALIBRATED"
    cv2.putText(banner, "REAL-TIME OBJECT DIMENSION MEASURER (OPENCV CONTOURS)", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 255), 2, lineType=cv2.LINE_AA)
    cv2.putText(banner, f"DETECTED OBJECTS: {len(measured_objects)} | {scale_str}", (15, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, lineType=cv2.LINE_AA)
                
    final_vis = np.vstack([banner, vis])
    edges_3c = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)
    edges_final = np.vstack([banner, edges_3c])
    
    return final_vis, edges_final, measured_objects, pixels_per_mm


def process_single_image(image_path, output_dir="output", ref_width_mm=85.6):
    """
    Processes single image for dimension measurement.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
        
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to decode image: {image_path}")
        
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    
    start_time = time.time()
    annotated, edges_mask, measured_objs, scale = measure_object_dimensions(
        image, ref_width_mm=ref_width_mm
    )
    proc_time = round(time.time() - start_time, 4)
    
    # Save output annotated image
    out_img_path = os.path.join(output_dir, f"{base_name}_dimensions.jpg")
    cv2.imwrite(out_img_path, annotated)
    
    # Build 2-Panel Side-by-Side Comparison Montage
    target_h = 500
    aspect = annotated.shape[1] / float(annotated.shape[0])
    p1 = cv2.resize(annotated, (int(target_h * aspect), target_h), interpolation=cv2.INTER_AREA)
    p2 = cv2.resize(edges_mask, (int(target_h * aspect), target_h), interpolation=cv2.INTER_AREA)
    
    divider = np.zeros((target_h, 5, 3), dtype=np.uint8)
    divider[:] = (180, 180, 180)
    montage = np.hstack([p1, divider, p2])
    
    montage_path = os.path.join(output_dir, f"{base_name}_comparison.jpg")
    cv2.imwrite(montage_path, montage)
    
    # Save JSON telemetry report
    report = {
        "filename": os.path.basename(image_path),
        "total_objects_measured": len(measured_objs),
        "pixels_per_mm": round(float(scale), 4) if scale is not None else None,
        "processing_time_sec": proc_time,
        "objects": measured_objs,
        "output_files": {
            "annotated_dimensions": out_img_path,
            "comparison_montage": montage_path
        }
    }
    
    json_path = os.path.join(output_dir, f"{base_name}_dimension_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"\n[+] Measured Document '{os.path.basename(image_path)}' in {proc_time}s")
    print(f"  - Detected Objects: {len(measured_objs)} | Scale: {scale:.2f} px/mm")
    print(f"  - Output Image: '{out_img_path}'")
    print(f"  - Metadata Report: '{json_path}'")
    
    return report


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Real-Time Object Dimension Measurer using OpenCV & Contours."
    )
    parser.add_argument(
        "-i", "--input", type=str, default="input/sample_objects_table.jpg",
        help="Path to input image file or directory of images."
    )
    parser.add_argument(
        "-o", "--output", type=str, default="output",
        help="Directory to save measured output images and JSON reports."
    )
    parser.add_argument(
        "--ref-width", type=float, default=85.6,
        help="Known physical width of reference object in mm (default: 85.6 mm for Credit Card)."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Auto-generate synthetic test image if input missing
    if not os.path.exists(args.input):
        print(f"[!] Input image '{args.input}' not found. Generating synthetic test bench image...")
        from generate_demo_objects import generate_synthetic_dimension_test_image
        args.input = generate_synthetic_dimension_test_image(output_path="input/sample_objects_table.jpg")
        
    print("\n==========================================================")
    print("  [DIM] REAL-TIME OBJECT DIMENSION MEASURER")
    print("  --------------------------------------------------------")
    print(f"  Input Source: {args.input}")
    print(f"  Output Dir  : {args.output}")
    print(f"  Ref Width   : {args.ref_width} mm")
    print("==========================================================")
    
    if os.path.isfile(args.input):
        process_single_image(args.input, output_dir=args.output, ref_width_mm=args.ref_width)
    elif os.path.isdir(args.input):
        imgs = glob.glob(os.path.join(args.input, "*.jpg")) + glob.glob(os.path.join(args.input, "*.png"))
        for img_p in sorted(imgs):
            process_single_image(img_p, output_dir=args.output, ref_width_mm=args.ref_width)


if __name__ == "__main__":
    main()
