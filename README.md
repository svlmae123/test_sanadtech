# Swimming Pool Detection from Aerial Images

**Author:** Asbai Salma  
**Project:** PFE Technical Test - Sanadtech  
**Date:** January 2026  
**Email:** salmaasbai91@gmail.com

## Overview

This project implements an automated swimming pool detection system for aerial and satellite images. The script accurately identifies and outlines swimming pools of any shape (rectangular, oval, irregular) using computer vision techniques.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip

### Setup Instructions

1. **Clone the repository**
```bash
   git clone https://github.com/YOUR-USERNAME/test_sanadtech.git
   cd test_sanadtech
```

2. **Install dependencies**
```bash
   pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python detect_pool.py --input input/image1.jpg
```

### Options

- `--input` or `-i` : Path to input aerial image (required)
- `--output` or `-o` : Output directory (default: 'output')
- `--quiet` or `-q` : Suppress detailed analysis output

### Examples
```bash
# Detect pools in a specific image
python detect_pool.py --input input/image1.jpg

# Save results to custom directory
python detect_pool.py --input input/image2.jpg --output results

# Run in quiet mode
python detect_pool.py --input input/image3.jpg --quiet
```

## Output Files

The script generates two files in the output directory:

### 1. `coordinates.txt`
Contains the detected pool boundary coordinates with detailed information:
- Pool area in pixels
- Boundary coordinates (x, y) for each point
- Multiple pools are numbered sequentially

Example format:
```
Pool #1
Area: 1635.50 pixels
Boundary Coordinates (x, y):
Point 1: (245, 189)
Point 2: (312, 187)
...
```

### 2. `output_image.jpg`
The input image with detected swimming pools outlined in **blue**.

## Sample Outputs

The repository includes sample inputs and outputs for demonstration:

### Sample Input Images (`input/`)
- `image1.jpg` - Residential area with rectangular pool
- `image2.jpg` - Multiple pools in aerial view
- `image3.jpg` - Oval-shaped pool
- `image4.jpg` - Irregular pool shape
- `image5.jpg` - Pool with varied lighting

### Sample Output (`output/`)
- `coordinates.txt` - Example coordinate file
- `output_image.jpg` - Example annotated image with blue pool outlines

## Technical Approach

### Detection Pipeline

1. **Color-based Segmentation**
   - HSV color space analysis
   - Multi-range detection for different water colors (cyan, blue, turquoise)
   - Robust to lighting variations

2. **Morphological Cleaning**
   - Noise removal
   - Gap filling within pool regions

3. **Geometric Filtering**
   - Area constraints (0.1% - 20% of image)
   - Compactness analysis (shape regularity)
   - Aspect ratio filtering (elongation check)
   - Solidity measurement (convexity)

4. **Boundary Refinement**
   - Edge detection on pool regions
   - Contour smoothing and simplification
   - Precise coordinate extraction

5. **Visualization**
   - Red outline drawing on original image
   - High-quality JPEG output

### Key Features

✅ Detects pools of any shape (rectangular, oval, irregular)  
✅ Handles multiple pools in a single image  
✅ Robust to varying lighting conditions  
✅ Detailed feedback during detection process  
✅ Fast processing (< 1 second per image on standard hardware)  

## Project Structure
```
test_sanadtech/
├── detect_pool.py          # Main detection script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .gitignore             # Git ignore rules
├── input/                 # Sample input images
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── image3.jpg
│   ├── image4.jpg
│   └── image5.jpg
└── output/                # Sample detection results
    ├── coordinates.txt    # Example coordinates
    └── output_image.jpg   # Example annotated image
```

## Testing

The script has been tested on various aerial images with:
- Different pool shapes (rectangular, oval, kidney-shaped, irregular)
- Various image resolutions
- Different lighting conditions
- Complex backgrounds

## Troubleshooting

**Issue:** No pools detected
- **Solution:** Check if pool water is visible and has blue/turquoise color

**Issue:** False positives
- **Solution:** Adjust geometric filters in the script (increase minimum area threshold)

**Issue:** Pool partially detected
- **Solution:** Ensure pool edges are clearly visible in the image

## Dependencies

- `opencv-python` - Image processing and computer vision
- `numpy` - Numerical operations
- `Pillow` - Image handling

Full list in `requirements.txt`
