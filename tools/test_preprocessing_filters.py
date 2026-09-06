#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to demonstrate all implemented preprocessing filters.
"""

import numpy as np
import sys
from pathlib import Path

# Add hcpa to path
sys.path.insert(0, str(Path(__file__).parent))

from utils_preprocessing import (
    get_filter,
    parse_filter_name,
    IdentityFilter,
    AHEFilter,
    CLAHEFilter,
    GammaFilter,
    MaxGreenFilter,
    GreenChannelExtraction,
    GrayscaleConversion,
    HistogramEqualization,
    BenGrahamNormalization,
    LABNormalization,
    GaussianBlur,
    MedianFilter,
    MultiScaleRetinex,
    OtsuThresholding,
    CannyEdgeDetection,
    FrangiVesselness,
    MorphologicalOperations,
    CompositeFilter
)


def create_test_image():
    """Create a synthetic test image (simulating retinal image)."""
    # Create a 256x256 RGB image with some patterns
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # Add gradient
    for i in range(256):
        image[i, :] = np.clip(50 + i, 0, 255)
    
    # Add some circles (simulating retinal features)
    center_x, center_y = 128, 128
    radius = 50
    for i in range(256):
        for j in range(256):
            if (i - center_x)**2 + (j - center_y)**2 < radius**2:
                image[i, j] = [200, 150, 100]
    
    return image


def test_single_filters():
    """Test individual filters."""
    print("\n" + "="*80)
    print("TESTING SINGLE FILTERS")
    print("="*80)
    
    image = create_test_image()
    
    test_cases = [
        ("baseline", {}),
        ("AHE40.0", {"clip_limit": 40.0, "tile_grid_size": (8, 8)}),
        ("CLAHE4.0", {"clip_limit": 4.0, "tile_grid_size": (8, 8)}),
        ("Gamma0.8", {"gamma": 0.8}),
        ("Gamma1.2", {"gamma": 1.2}),
        ("MaxGreen2.0", {"max_green_factor": 2.0}),
        ("MaxGreen4.0", {"max_green_factor": 4.0}),
        ("green_channel", {}),
        ("grayscale", {}),
        ("histogram_equalization", {}),
        ("ben_graham_norm", {}),
        ("lab_norm", {}),
        ("gaussian_blur", {"sigma": 5.0}),
        ("median_filter", {"kernel_size": 5}),
        ("multi_scale_retinex", {}),
        ("otsu_threshold", {}),
        ("canny_edge", {"lower_threshold": 100, "upper_threshold": 200}),
        ("morphological", {"kernel_size": 5, "operation": "open"}),
    ]
    
    for filter_name, params in test_cases:
        try:
            filter_obj = get_filter(filter_name, params)
            result = filter_obj.apply(image)
            print(f"✓ {filter_name:30s} -> Shape: {result.shape}, dtype: {result.dtype}")
        except Exception as e:
            print(f"✗ {filter_name:30s} -> ERROR: {e}")


def test_composite_filters():
    """Test composite filters."""
    print("\n" + "="*80)
    print("TESTING COMPOSITE FILTERS")
    print("="*80)
    
    image = create_test_image()
    
    composite_filters = [
        "AHE40.0_CLAHE4.0",
        "AHE40.0_CLAHE4.0_Gamma0.8",
        "AHE40.0_CLAHE4.0_Gamma0.8_MaxGreen2.0",
        "AHE40.0_CLAHE4.0_Gamma0.8_MaxGreen4.0",
        "AHE40.0_CLAHE4.0_Gamma1.2_MaxGreen2.0",
        "CLAHE4.0_Gamma0.8",
        "CLAHE4.0_Gamma1.2",
    ]
    
    for filter_name in composite_filters:
        try:
            filter_obj = get_filter(filter_name, {})
            result = filter_obj.apply(image)
            print(f"✓ {filter_name:50s} -> Shape: {result.shape}, dtype: {result.dtype}")
        except Exception as e:
            print(f"✗ {filter_name:50s} -> ERROR: {e}")


def test_parse_filter_name():
    """Test filter name parsing."""
    print("\n" + "="*80)
    print("TESTING FILTER NAME PARSING")
    print("="*80)
    
    test_names = [
        "baseline",
        "AHE40.0",
        "CLAHE4.0",
        "Gamma0.8",
        "MaxGreen2.0",
        "AHE40.0_CLAHE4.0_Gamma0.8_MaxGreen2.0",
    ]
    
    for name in test_names:
        try:
            parsed_name, params = parse_filter_name(name)
            print(f"✓ {name:50s}")
            print(f"  Parsed name: {parsed_name}")
            print(f"  Parameters: {params}")
            print()
        except Exception as e:
            print(f"✗ {name:50s} -> ERROR: {e}\n")


def test_filter_matrix():
    """Test all filters from the filter matrix template."""
    print("\n" + "="*80)
    print("TESTING FILTERS FROM FILTER MATRIX")
    print("="*80)
    
    image = create_test_image()
    
    filter_matrix = [
        {"name": "baseline", "params": {}},
        {"name": "AHE40.0", "params": {"clip_limit": 40.0, "tile_grid_size": (8, 8)}},
        {"name": "AHE50.0", "params": {"clip_limit": 50.0, "tile_grid_size": (8, 8)}},
        {"name": "CLAHE2.0", "params": {"clip_limit": 2.0, "tile_grid_size": (8, 8)}},
        {"name": "CLAHE4.0", "params": {"clip_limit": 4.0, "tile_grid_size": (8, 8)}},
        {"name": "Gamma0.8", "params": {"gamma": 0.8}},
        {"name": "Gamma1.2", "params": {"gamma": 1.2}},
        {"name": "MaxGreen2.0", "params": {"max_green_factor": 2.0}},
        {"name": "MaxGreen3.0", "params": {"max_green_factor": 3.0}},
        {"name": "MaxGreen4.0", "params": {"max_green_factor": 4.0}},
    ]
    
    for filter_spec in filter_matrix:
        try:
            filter_obj = get_filter(filter_spec["name"], filter_spec["params"])
            result = filter_obj.apply(image)
            config = filter_obj.get_config()
            print(f"✓ {filter_spec['name']:25s} -> Shape: {result.shape}")
        except Exception as e:
            print(f"✗ {filter_spec['name']:25s} -> ERROR: {e}")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("PREPROCESSING FILTERS TEST SUITE")
    print("="*80)
    
    test_single_filters()
    test_composite_filters()
    test_parse_filter_name()
    test_filter_matrix()
    
    print("\n" + "="*80)
    print("TEST SUITE COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
