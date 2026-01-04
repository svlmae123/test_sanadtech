#!/usr/bin/env python3
"""
Pool Detection Script for Aerial Images 
"""

import cv2
import numpy as np
import argparse
import sys
from pathlib import Path


def detect_pool_water(img):
    """
    Detect pool water with balanced color filtering
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_cyan = np.array([80, 60, 80])
    upper_cyan = np.array([100, 255, 255])
    mask_cyan = cv2.inRange(hsv, lower_cyan, upper_cyan)
    
    lower_blue = np.array([95, 80, 90])
    upper_blue = np.array([110, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    lower_light = np.array([85, 40, 150])
    upper_light = np.array([100, 150, 255])
    mask_light = cv2.inRange(hsv, lower_light, upper_light)
    
    mask = cv2.bitwise_or(mask_cyan, mask_blue)
    mask = cv2.bitwise_or(mask, mask_light)
    
    return mask


def clean_mask(mask):
    """
    Clean mask with careful morphology
    """
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium, iterations=1)
    
    return mask


def filter_pool_contours(contours, img_shape, verbose=True):
    """
    Apply balanced geometric filters with detailed feedback
    """
    image_area = img_shape[0] * img_shape[1]
    valid_pools = []
    
    if verbose:
        print(f"\n{'='*60}")
        print("Analyzing contours...")
        print(f"{'='*60}")
    
    for idx, contour in enumerate(contours, 1):
        area = cv2.contourArea(contour)
        
        min_area = image_area * 0.001   
        max_area = image_area * 0.20    
        
        if verbose:
            print(f"\nContour #{idx}:")
            print(f"  Area: {area:.2f} pixels ({area/image_area*100:.2f}% of image)")
        
        if area < min_area:
            if verbose:
                print(f"  REJECTED: Too small (min: {min_area:.2f})")
            continue
        
        if area > max_area:
            if verbose:
                print(f"  REJECTED: Too large (max: {max_area:.2f})")
            continue
        
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            if verbose:
                print(f"  REJECTED: Invalid perimeter")
            continue
        
        compactness = 4 * np.pi * area / (perimeter * perimeter)
        if verbose:
            print(f"  Compactness: {compactness:.3f} (circle=1.0, square=0.785)")
        
        if compactness < 0.20:   
            if verbose:
                print(f"  REJECTED: Not compact enough (min: 0.20)")
            continue
        
        # Aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        if verbose:
            print(f"  Aspect ratio: {aspect_ratio:.2f} (1.0=square)")
        
        if aspect_ratio > 5:  
            if verbose:
                print(f"  REJECTED: Too elongated (max: 5)")
            continue
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            if verbose:
                print(f"  Solidity: {solidity:.3f} (1.0=perfect convex)")
            
            if solidity < 0.60:  
                if verbose:
                    print(f"  REJECTED: Too irregular (min: 0.60)")
                continue
        
        # All checks passed!
        if verbose:
            print(f"  ACCEPTED as pool")
        
        valid_pools.append(contour)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Result: {len(valid_pools)} pool(s) validated")
        print(f"{'='*60}\n")
    
    return valid_pools


def refine_pool_boundary(contour, img, mask):
    """
    Refine pool boundary using edge detection
    """
    x, y, w, h = cv2.boundingRect(contour)
    padding = 10
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img.shape[1], x + w + padding)
    y2 = min(img.shape[0], y + h + padding)
    
    roi_mask = mask[y1:y2, x1:x2]
    
    roi_contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if roi_contours:

        refined = max(roi_contours, key=cv2.contourArea)
        
        refined = refined + np.array([x1, y1])
        
        epsilon = 0.002 * cv2.arcLength(refined, True)
        refined = cv2.approxPolyDP(refined, epsilon, True)
        
        return refined
    
    epsilon = 0.003 * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)


def detect_pools(image_path, output_dir="output", verbose=True):
    """
    Detect swimming pools with balanced precision
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image from {image_path}")
        return False
    
    print(f"Processing image: {image_path}")
    print(f"Image shape: {img.shape}")
    print(f"Image area: {img.shape[0] * img.shape[1]} pixels")
    
    # Step 1: Detect pool water
    mask = detect_pool_water(img)
    
    # Step 2: Clean mask
    mask_clean = clean_mask(mask)
    
    # Step 3: Find contours with maximum detail
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        print("Warning: No pools detected in the image")
        return False
    
    print(f"Found {len(contours)} initial contours")
    
    # Step 4: Filter contours
    valid_pools = filter_pool_contours(contours, img.shape, verbose=verbose)
    
    if not valid_pools:
        print("\n  No valid pools found. Try:")
        print("   1. Check debug_mask.jpg - is the pool water white?")
        print("   2. If pool is missing: color detection needs adjustment")
        print("   3. If pool is there but rejected: filters are too strict")
        return False
    
    valid_pools = sorted(valid_pools, key=cv2.contourArea, reverse=True)
    
    print(f" {len(valid_pools)} pool(s) detected successfully!")
    
    # Step 5: Refine boundaries
    refined_pools = []
    for idx, contour in enumerate(valid_pools):
        if verbose:
            print(f"Refining pool #{idx+1} boundary...")
        refined = refine_pool_boundary(contour, img, mask_clean)
        refined_pools.append(refined)
    
    # Step 6: Save coordinates
    coords_file = Path(output_dir) / "coordinates.txt"
    with open(coords_file, 'w') as f:
        f.write(f"Swimming Pool Detection Results\n")
        f.write(f"Image: {Path(image_path).name}\n")
        f.write(f"Total Pools Detected: {len(refined_pools)}\n")
        f.write("=" * 60 + "\n\n")
        
        for pool_idx, contour in enumerate(refined_pools, 1):
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            x, y, w, h = cv2.boundingRect(contour)
            
            f.write(f"Pool #{pool_idx}\n")
            f.write(f"Area: {area:.2f} square pixels\n")
            f.write(f"Perimeter: {perimeter:.2f} pixels\n")
            f.write(f"Bounding Box: x={x}, y={y}, width={w}, height={h}\n")
            f.write(f"Boundary Coordinates (x, y):\n")
            f.write("-" * 50 + "\n")
            
            coordinates = contour.squeeze().tolist()
            
            if len(contour) > 2:
                if isinstance(coordinates[0], list):
                    for i, coord in enumerate(coordinates):
                        f.write(f"Point {i+1}: ({coord[0]}, {coord[1]})\n")
                else:
                    f.write(f"Point 1: ({coordinates[0]}, {coordinates[1]})\n")
            
            f.write("\n")
    
    print(f"Coordinates saved to: {coords_file}")
    
    # Step 7: Draw contours
    output_img = img.copy()
    
    for pool_idx, contour in enumerate(refined_pools, 1):
        cv2.drawContours(output_img, [contour], -1, (255, 0, 0), 1)
        
        area = cv2.contourArea(contour)
        print(f"Pool #{pool_idx}: Area = {area:.2f} pixels")
    
    # Save output
    output_image_file = Path(output_dir) / "output_image.jpg"
    cv2.imwrite(str(output_image_file), output_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Output image saved to: {output_image_file}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Detect swimming pools with balanced precision and detailed feedback',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python detect_pool.py image.jpg
  python detect_pool.py image.jpg --output results
  python detect_pool.py image.jpg --quiet

Features:
- Multi-range color detection for different water conditions
- Balanced geometric filtering
- Detailed feedback showing why contours are accepted/rejected
- Debug mask output for troubleshooting
        """
    )
    
    parser.add_argument('image', help='Path to input aerial image')
    parser.add_argument('--output', '-o', default='output',
                       help='Output directory (default: output)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress detailed analysis output')
    
    args = parser.parse_args()
    
    if not Path(args.image).exists():
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)
    
    success = detect_pools(args.image, args.output, verbose=not args.quiet)
    
    if success:
        print("\n Pool detection completed successfully!")
        print(f" Check the '{args.output}' directory for results")
        sys.exit(0)
    else:
        print("\n Pool detection failed")
        sys.exit(1)


if __name__ == "__main__":
    main()