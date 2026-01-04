#!/usr/bin/env python3
"""
Pool Detection Script for Aerial Images
Detects swimming pools and generates coordinates and outlined image
Optimized to reduce false positives
"""

import cv2
import numpy as np
import argparse
import sys
from pathlib import Path


def detect_pools(image_path, output_dir="output"):
    """
    Detect swimming pools in aerial image with strict filtering
    
    Args:
        image_path: Path to input aerial image
        output_dir: Directory for output files
    
    Returns:
        bool: True if at least one pool detected successfully
    """
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image from {image_path}")
        return False
    
    print(f"Processing image: {image_path}")
    print(f"Image shape: {img.shape}")
    
    # Convert to different color spaces
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Define strict blue/cyan color ranges for pool water
    # Focused on bright, saturated blue/cyan typical of pool water
    lower_cyan = np.array([85, 80, 80])    # More saturated
    upper_cyan = np.array([100, 255, 255])
    
    lower_blue = np.array([95, 100, 100])  # Even more strict
    upper_blue = np.array([115, 255, 255])
    
    # Create masks
    mask_cyan = cv2.inRange(hsv, lower_cyan, upper_cyan)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.bitwise_or(mask_cyan, mask_blue)
    
    # Morphological operations - LESS aggressive to keep pools separated
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_medium = np.ones((5, 5), np.uint8)
    
    # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    # Fill small gaps WITHIN pools but don't merge separate pools
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Warning: No pools detected in the image")
        return False
    
    # STRICT filtering criteria for pools
    image_area = img.shape[0] * img.shape[1]
    min_area = image_area * 0.002   # At least 0.2% of image
    max_area = image_area * 0.15    # At most 15% of image
    
    valid_pools = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Check area
        if area < min_area or area > max_area:
            continue
        
        # Check shape compactness (pools are usually compact shapes)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        
        compactness = 4 * np.pi * area / (perimeter * perimeter)
        # Pools typically have compactness > 0.3 (rectangles ~0.785, circles = 1.0)
        if compactness < 0.25:
            continue
        
        # Check aspect ratio (pools shouldn't be too elongated)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        if aspect_ratio > 4:  # Too elongated, probably not a pool
            continue
        
        # Check if shape is reasonably convex (pools don't have complex shapes)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            if solidity < 0.7:  # Too irregular
                continue
        
        valid_pools.append(contour)
    
    if not valid_pools:
        print("Warning: No valid pool contours found after filtering")
        return False
    
    # Sort by area (largest first)
    valid_pools = sorted(valid_pools, key=cv2.contourArea, reverse=True)
    
    print(f"Number of pools detected: {len(valid_pools)}")
    
    # Save coordinates for ALL detected pools
    coords_file = Path(output_dir) / "coordinates.txt"
    with open(coords_file, 'w') as f:
        f.write(f"Swimming Pool Detection Results\n")
        f.write(f"Image: {Path(image_path).name}\n")
        f.write(f"Total Pools Detected: {len(valid_pools)}\n")
        f.write("=" * 60 + "\n\n")
        
        for pool_idx, contour in enumerate(valid_pools, 1):
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
            
            # Handle both single and multiple points
            if len(contour) > 2:
                if isinstance(coordinates[0], list):
                    for i, coord in enumerate(coordinates):
                        f.write(f"Point {i+1}: ({coord[0]}, {coord[1]})\n")
                else:
                    f.write(f"Point 1: ({coordinates[0]}, {coordinates[1]})\n")
            
            f.write("\n")
    
    print(f"Coordinates saved to: {coords_file}")
    
    # Draw blue outline on image for ALL pools
    output_img = img.copy()
    
    for pool_idx, contour in enumerate(valid_pools, 1):
        # Draw thin blue contour (BGR format: Blue = 255, Green = 0, Red = 0)
        # Thickness = 1 pixel for a fine outline matching the example
        cv2.drawContours(output_img, [contour], -1, (255, 0, 0), 1)
        
        area = cv2.contourArea(contour)
        print(f"Pool #{pool_idx}: Area = {area:.2f} pixels")
    
    # Save output image
    output_image_file = Path(output_dir) / "output_image.jpg"
    cv2.imwrite(str(output_image_file), output_img)
    print(f"Output image saved to: {output_image_file}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Detect swimming pools in aerial images with high accuracy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python detect_pool.py image.jpg
  python detect_pool.py image.jpg --output results
  
The script uses strict filtering to minimize false positives:
- Color-based detection (cyan/blue water)
- Size filtering (realistic pool dimensions)
- Shape analysis (compactness, aspect ratio, solidity)
        """
    )
    
    parser.add_argument('image', help='Path to input aerial image')
    parser.add_argument('--output', '-o', default='output',
                       help='Output directory (default: output)')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not Path(args.image).exists():
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)
    
    # Detect pools
    success = detect_pools(args.image, args.output)
    
    if success:
        print("\n✅ Pool detection completed successfully!")
        print(f"📁 Check the '{args.output}' directory for results")
        sys.exit(0)
    else:
        print("\n❌ Pool detection failed")
        sys.exit(1)


if __name__ == "__main__":
    main()