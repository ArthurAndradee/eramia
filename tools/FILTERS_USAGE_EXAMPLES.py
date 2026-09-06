#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Usage Examples for Preprocessing Filters

Copy-paste examples showing how to use the implemented filters.
"""

from utils_preprocessing import get_filter, parse_filter_name
import cv2
import numpy as np

# ============================================================================
# Example 1: Single Filter with Default Parameters
# ============================================================================

def example_single_filter():
    """Apply a single filter to an image."""
    print("Example 1: Single Filter")
    
    # Load image
    image = cv2.imread("path/to/retinal/image.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create and apply CLAHE filter
    clahe_filter = get_filter("CLAHE4.0", {"clip_limit": 4.0, "tile_grid_size": (8, 8)})
    filtered = clahe_filter.apply(image)
    
    # Get configuration for metadata
    config = clahe_filter.get_config()
    print(f"Filter: {config['name']}")
    print(f"Params: {config['params']}")
    
    return filtered


# ============================================================================
# Example 2: Composite Filter (Automatic Parsing)
# ============================================================================

def example_composite_filter():
    """Apply a composite filter (multiple filters in sequence)."""
    print("\nExample 2: Composite Filter")
    
    # Load image
    image = cv2.imread("path/to/retinal/image.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create composite filter (automatically parsed)
    # Applies: AHE(40) → CLAHE(4) → Gamma(0.8) → MaxGreen(2)
    composite = get_filter("AHE40.0_CLAHE4.0_Gamma0.8_MaxGreen2.0", {})
    filtered = composite.apply(image)
    
    config = composite.get_config()
    print(f"Filter chain: {config['name']}")
    print(f"Sub-filters: {len(config['filters'])}")
    
    return filtered


# ============================================================================
# Example 3: Parse Filter Name to Extract Parameters
# ============================================================================

def example_parse_filter_name():
    """Parse filter name to get structured parameters."""
    print("\nExample 3: Parse Filter Name")
    
    filter_name = "AHE40.0_CLAHE4.0_Gamma0.8_MaxGreen2.0"
    name, params = parse_filter_name(filter_name)
    
    print(f"Filter name: {name}")
    print(f"Parameters: {params}")
    
    return name, params


# ============================================================================
# Example 4: Apply All Filters in Filter Matrix
# ============================================================================

def example_filter_matrix():
    """Apply all filters from the filter matrix."""
    print("\nExample 4: Filter Matrix")
    
    # Load image
    image = cv2.imread("path/to/retinal/image.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Filter matrix (from filter_matrix_template.json)
    filters = [
        "baseline",
        "AHE40.0",
        "AHE50.0",
        "CLAHE2.0",
        "CLAHE4.0",
        "Gamma0.8",
        "Gamma1.2",
        "MaxGreen2.0",
        "MaxGreen3.0",
        "AHE40.0_CLAHE4.0",
        "AHE40.0_CLAHE4.0_Gamma0.8",
        "AHE40.0_CLAHE4.0_Gamma0.8_MaxGreen2.0",
        "AHE40.0_CLAHE4.0_Gamma0.8_MaxGreen4.0",
        "CLAHE4.0_Gamma0.8",
        "CLAHE4.0_Gamma1.2",
    ]
    
    results = {}
    for filter_name in filters:
        try:
            f = get_filter(filter_name, {})
            results[filter_name] = f.apply(image)
            print(f"✓ {filter_name}")
        except Exception as e:
            print(f"✗ {filter_name}: {e}")
    
    return results


# ============================================================================
# Example 5: Advanced Filters
# ============================================================================

def example_advanced_filters():
    """Apply advanced preprocessing filters."""
    print("\nExample 5: Advanced Filters")
    
    # Load image
    image = cv2.imread("path/to/retinal/image.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Green channel extraction
    green = get_filter("green_channel", {}).apply(image)
    print(f"Green channel shape: {green.shape}")
    
    # Gaussian blur
    blurred = get_filter("gaussian_blur", {"sigma": 5.0}).apply(image)
    print(f"Gaussian blur shape: {blurred.shape}")
    
    # Multi-Scale Retinex (illumination correction)
    retinex = get_filter("multi_scale_retinex", {}).apply(image)
    print(f"Retinex shape: {retinex.shape}")
    
    # Morphological operations
    morph = get_filter("morphological", {"kernel_size": 5, "operation": "open"}).apply(image)
    print(f"Morphological shape: {morph.shape}")
    
    return {
        "green": green,
        "blurred": blurred,
        "retinex": retinex,
        "morph": morph
    }


# ============================================================================
# Example 6: Using Filters in Script 1 (Dataset Creation)
# ============================================================================

def example_script1_usage():
    """Example of how to use filters in Script 1 (dataset creation)."""
    print("\nExample 6: Script 1 Usage Pattern")
    
    filter_name = "AHE40.0_CLAHE4.0_Gamma0.8_MaxGreen2.0"
    
    # Parse filter name
    name, params = parse_filter_name(filter_name)
    
    # Create filter
    filter_obj = get_filter(filter_name, {})
    
    # Process images in a directory
    import os
    from pathlib import Path
    
    image_dir = Path("path/to/dataset/train")
    output_dir = Path("path/to/filtered_dataset/train")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for image_file in image_dir.glob("*.jpg"):
        # Load image
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply filter
        filtered = filter_obj.apply(image)
        
        # Save filtered image
        filtered_bgr = cv2.cvtColor(filtered, cv2.COLOR_RGB2BGR)
        output_path = output_dir / image_file.name
        cv2.imwrite(str(output_path), filtered_bgr)
        
        print(f"Processed: {image_file.name}")
    
    # Save filter metadata
    metadata = {
        "filter_name": filter_name,
        "filter_config": filter_obj.get_config(),
        "parameters": params
    }
    
    import json
    with open(output_dir.parent / "filter_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to filter_metadata.json")


# ============================================================================
# Example 7: Custom Filter Combinations
# ============================================================================

def example_custom_combinations():
    """Create custom filter combinations."""
    print("\nExample 7: Custom Filter Combinations")
    
    # Load image
    image = cv2.imread("path/to/retinal/image.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Define custom combinations
    combinations = [
        "Gamma0.8",                          # Just brighten
        "MaxGreen3.0",                       # Just suppress green
        "CLAHE4.0_Gamma0.8",                # CLAHE then gamma
        "AHE40.0_MaxGreen2.0",              # AHE then green suppression
        "CLAHE4.0_Gamma1.2_MaxGreen3.0",   # Three-filter chain
    ]
    
    for combo in combinations:
        try:
            f = get_filter(combo, {})
            result = f.apply(image)
            print(f"✓ {combo:40s} → shape {result.shape}")
        except Exception as e:
            print(f"✗ {combo:40s} → {e}")


# ============================================================================
# Example 8: Filter Factory and Direct Object Creation
# ============================================================================

def example_filter_factory():
    """Use the factory function to create filters."""
    print("\nExample 8: Filter Factory Patterns")
    
    image = cv2.imread("path/to/retinal/image.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Method 1: Using factory function with filter name
    f1 = get_filter("CLAHE4.0", {"clip_limit": 4.0, "tile_grid_size": (8, 8)})
    result1 = f1.apply(image)
    print(f"Factory (CLAHE): {result1.shape}")
    
    # Method 2: Composite filter (auto-parsed from name)
    f2 = get_filter("AHE40.0_CLAHE4.0_Gamma0.8", {})
    result2 = f2.apply(image)
    print(f"Factory (Composite): {result2.shape}")
    
    # Method 3: Direct class instantiation (not recommended)
    from utils_preprocessing import CLAHEFilter
    f3 = CLAHEFilter("CLAHE4.0", {"clip_limit": 4.0, "tile_grid_size": (8, 8)})
    result3 = f3.apply(image)
    print(f"Direct (CLAHEFilter): {result3.shape}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Preprocessing Filters - Usage Examples")
    print("="*80)
    
    # Uncomment examples to run:
    
    # example_single_filter()
    # example_composite_filter()
    # example_parse_filter_name()
    # example_filter_matrix()
    # example_advanced_filters()
    # example_script1_usage()
    # example_custom_combinations()
    # example_filter_factory()
    
    print("\n" + "="*80)
    print("Copy these examples and adapt them to your use case!")
    print("="*80)
