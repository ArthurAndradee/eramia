# Quick Reference: Preprocessing Filters

## All Available Filters

### Baseline
```python
filter_obj = get_filter("baseline", {})
```

### Core Research Filters
| Filter | Name | Parameters |
|--------|------|------------|
| Adaptive Histogram Equalization | AHE40.0, AHE50.0 | clip_limit: 40.0/50.0 |
| Contrast Limited AHE | CLAHE2.0, CLAHE4.0 | clip_limit: 2.0/4.0 |
| Gamma Correction | Gamma0.8, Gamma1.2 | gamma: 0.8/1.2 |
| Green Channel Suppression | MaxGreen2.0, MaxGreen3.0, MaxGreen4.0 | max_green_factor: 2.0/3.0/4.0 |

### Advanced Preprocessing Filters
| Filter | Name | Parameters |
|--------|------|------------|
| Green Channel Extraction | green_channel | - |
| Grayscale Conversion | grayscale | - |
| Histogram Equalization | histogram_equalization | - |
| Ben Graham Normalization | ben_graham_norm | - |
| LAB Normalization | lab_norm | - |
| Gaussian Blur | gaussian_blur | sigma: 5.0-10.0 |
| Median Filter | median_filter | kernel_size: 3-5 |
| Multi-Scale Retinex | multi_scale_retinex | - |
| Otsu Thresholding | otsu_threshold | - |
| Canny Edge Detection | canny_edge | lower_threshold, upper_threshold |
| Frangi Vesselness | frangi_vesselness | scales, beta, c |
| Morphological Operations | morphological | kernel_size, operation |

## Quick Usage

### Single Filter
```python
from utils_preprocessing import get_filter

# With parameters
f = get_filter("CLAHE4.0", {"clip_limit": 4.0, "tile_grid_size": (8, 8)})
result = f.apply(image)

# With defaults
f = get_filter("Gamma0.8", {"gamma": 0.8})
result = f.apply(image)
```

### Composite Filter
```python
# Automatically parsed - applies left to right
f = get_filter("AHE40.0_CLAHE4.0_Gamma0.8_MaxGreen2.0", {})
result = f.apply(image)
```

### Parse Filter Name
```python
from utils_preprocessing import parse_filter_name

name, params = parse_filter_name("AHE40.0_CLAHE4.0_Gamma0.8")
```

## Filter Matrix (All Combinations)

### 5 Basic Filters
- baseline
- AHE40.0
- AHE50.0
- CLAHE2.0
- CLAHE4.0
- Gamma0.8
- Gamma1.2
- MaxGreen2.0
- MaxGreen3.0
- MaxGreen4.0

### 7 Composite Filters (from template)
1. AHE40.0_CLAHE4.0
2. AHE40.0_CLAHE4.0_Gamma0.8
3. AHE40.0_CLAHE4.0_Gamma0.8_MaxGreen2.0
4. AHE40.0_CLAHE4.0_Gamma0.8_MaxGreen4.0
5. AHE40.0_CLAHE4.0_Gamma1.2_MaxGreen2.0
6. CLAHE4.0_Gamma0.8
7. CLAHE4.0_Gamma1.2

## CSV Parameters Used

| From CSV | Implemented As | Value Range |
|----------|-----------------|------------|
| CLAHE clip_limit | CLAHE2.0, CLAHE4.0 | 2.0-5.5 |
| CLAHE tile_grid | tile_grid_size | (8,8) or (32,32) |
| Gamma (brighten) | Gamma0.8 | 0.8 |
| Gamma (darken) | Gamma1.2 | 1.2 |
| Green suppression | MaxGreen2.0-4.0 | 2.0-4.0 |
| Gaussian sigma | GaussianBlur | 5.0-10.0 |
| Median kernel | MedianFilter | 3×3 or 5×5 |
| Canny lower | CannyEdgeDetection | 6-10 |
| Canny upper | CannyEdgeDetection | 32-60 |
| Frangi scales | FrangiVesselness | 1-15 |
| Frangi β | FrangiVesselness | 5 |
| Morpho kernel | MorphologicalOperations | 5×5, 11×11, 23×23 |

## Integration Points

### Script 1 (Dataset Creation)
```python
# Create dataset with specific filter
filtro = "AHE40.0_CLAHE4.0_Gamma0.8_MaxGreen2.0"
filter_obj = get_filter(filtro, {})

for image in dataset:
    filtered_image = filter_obj.apply(image)
    save_image(filtered_image)
```

### Script 2 (Training)
Images already filtered in Script 1, so no additional filtering needed.

### Script 3 (Evaluation)
Evaluation uses the filtered images from Script 1.

## Key Points
1. **Composite filters** are automatically parsed from underscore-separated names
2. **All filters** return uint8 images [0-255]
3. **RGB images** are properly handled with color space conversions
4. **Default parameters** work well for most use cases
5. **get_config()** returns metadata for logging

## Examples Location
See `FILTERS_USAGE_EXAMPLES.py` for copy-paste ready code samples.

## Full Documentation
See `FILTERS_IMPLEMENTATION.md` for comprehensive API reference.
