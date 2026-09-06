# Preprocessing Filters Implementation

## Overview

Complete implementation of preprocessing filters for diabetic retinopathy image analysis, based on parameters from `parametros-filtros-dr.csv`. All filters are integrated into `utils_preprocessing.py`.

## Filter Classes Implemented

### 1. **IdentityFilter** (Baseline)
- **Purpose**: No-operation filter for control group
- **Parameters**: None
- **Usage**: `"baseline"`
- **CSV Reference**: Control group for comparative analysis

### 2. **AHEFilter** (Adaptive Histogram Equalization)
- **Purpose**: Enhance contrast locally while preserving overall brightness
- **Parameters**: 
  - `clip_limit` (float): Contrast limit factor (e.g., 40.0, 50.0)
  - `tile_grid_size` (tuple): Size of grid tiles (default: (8, 8))
- **CSV Reference**: Not explicitly in CSV (superseded by CLAHE)
- **Implementation**: Converts to LAB colorspace, applies to L channel, converts back
- **Supported Names**: `AHE40.0`, `AHE50.0`

### 3. **CLAHEFilter** (Contrast Limited Adaptive Histogram Equalization)
- **Purpose**: More advanced contrast enhancement with limitations to prevent over-enhancement
- **Parameters**:
  - `clip_limit` (float): Clip limit (2.0-5.5 per CSV, we support any value)
  - `tile_grid_size` (tuple): Grid tiles (default: (8, 8), CSV mentions 32×32 variant)
- **CSV Reference**: "CLAHE, Clip limit / Tile grid size"
- **Implementation**: LAB colorspace enhancement
- **Supported Names**: `CLAHE2.0`, `CLAHE4.0`

### 4. **GammaFilter** (Gamma Correction)
- **Purpose**: Brighten (γ<1) or darken (γ>1) images
- **Parameters**:
  - `gamma` (float): Gamma value
    - γ < 1.0: Brighten image
    - γ = 1.0: No change
    - γ > 1.0: Darken image
- **CSV Reference**: "Correção Gamma" - Gamma adaptativo or fixed values
- **Implementation**: Power law transformation: output = input^(1/gamma)
- **Supported Names**: `Gamma0.8`, `Gamma1.2`

### 5. **MaxGreenFilter** (Green Channel Suppression)
- **Purpose**: Suppress green channel to enhance red/dark structures (blood vessels, lesions)
- **Parameters**:
  - `max_green_factor` (float): Division factor for green channel (e.g., 2.0, 3.0, 4.0)
- **CSV Reference**: "Extração do canal verde"
- **Implementation**: Divides green channel by factor
- **Supported Names**: `MaxGreen2.0`, `MaxGreen3.0`, `MaxGreen4.0`

### 6. **GreenChannelExtraction**
- **Purpose**: Extract only the green channel from RGB image
- **Parameters**: None
- **CSV Reference**: "Extração do canal verde - Canal G (RGB)"
- **Implementation**: Returns only channel 1 (green) from RGB image

### 7. **GrayscaleConversion**
- **Purpose**: Convert RGB to grayscale for single-channel processing
- **Parameters**: None
- **CSV Reference**: "Conversão para escala de cinza (Grayscale)"
- **Implementation**: Standard RGB→Gray conversion using fixed weights

### 8. **HistogramEqualization**
- **Purpose**: Global histogram equalization to enhance contrast globally
- **Parameters**: None
- **CSV Reference**: "Equalização de Histograma (global) - Não-paramétrico"
- **Implementation**: Equalizes L channel in LAB colorspace

### 9. **BenGrahamNormalization**
- **Purpose**: Standardize color appearance specific to retinal images
- **Parameters**: None
- **CSV Reference**: "Normalização/padronização de cor - Subtração da cor média local"
- **Implementation**: 
  - Subtract mean color
  - Divide by standard deviation
  - Rescale to [0, 255]
- **Reference**: Ben Graham, Kaggle DR Competition (2015)

### 10. **LABNormalization**
- **Purpose**: Normalize in LAB colorspace, particularly L (luminosity) channel
- **Parameters**: None
- **CSV Reference**: "Normalização no espaço LAB (canal de luminosidade)"
- **Implementation**: Min-max normalization of L channel in LAB space

### 11. **GaussianBlur**
- **Purpose**: Smooth image to reduce noise
- **Parameters**:
  - `sigma` (float): Standard deviation of Gaussian kernel
    - CSV default: σ = 5 pixels
    - CSV formula: σ = radius_target / 30 (e.g., 300px radius → σ ≈ 10)
- **CSV Reference**: "Filtro Gaussiano - Sigma (σ)"
- **Implementation**: Gaussian blur using cv2.GaussianBlur

### 12. **MedianFilter**
- **Purpose**: Noise reduction while preserving edges
- **Parameters**:
  - `kernel_size` (int): Size of median filter kernel
    - CSV options: 3×3 (light) or 5×5 (common)
- **CSV Reference**: "Filtro de Mediana - Tamanho do kernel"
- **Implementation**: cv2.medianBlur

### 13. **MultiScaleRetinex**
- **Purpose**: Illumination correction using multi-scale Retinex algorithm
- **Parameters**:
  - `scales` (list): Scales for multi-scale processing (default: [15, 80, 250])
- **CSV Reference**: "Correção de iluminação (Illumination/Shade Correction)"
- **Implementation**: 
  - Multi-scale Gaussian blur
  - Retinex: log(input) - log(gaussian)
  - Combines three scales with equal weights

### 14. **OtsuThresholding**
- **Purpose**: Automatic thresholding using Otsu method
- **Parameters**: None (fully automatic)
- **CSV Reference**: "Limiarização de Otsu - Totalmente automático"
- **Implementation**: cv2.threshold with OTSU flag

### 15. **CannyEdgeDetection**
- **Purpose**: Detect edges in image
- **Parameters**:
  - `lower_threshold` (int): Lower threshold (CSV: 6-10)
  - `upper_threshold` (int): Upper threshold (CSV: 32-60)
- **CSV Reference**: "Detecção de bordas de Canny"
- **Implementation**: cv2.Canny with configurable thresholds

### 16. **FrangiVesselness**
- **Purpose**: Vessel detection and enhancement using Frangi filter
- **Parameters**:
  - `scales` (list): Scales for multi-scale Hessian (CSV: 1-15)
  - `beta` (float): Sensitivity to plate-like structures (CSV: 5)
  - `c` (float): Sensitivity to noise (CSV: (1, 15))
- **CSV Reference**: "Filtro de Frangi (Vesselness) - Escala (σ) / parâmetros β, c"
- **Implementation**: 
  - Multi-scale Hessian matrix eigenvalue analysis
  - Frangi filter equation with tunable parameters

### 17. **MorphologicalOperations**
- **Purpose**: Morphological transformations (opening, closing, erosion, dilation)
- **Parameters**:
  - `kernel_size` (int): Size of morphological kernel (CSV: 5×5, 11×11, 23×23)
  - `operation` (str): Type of operation
    - `"open"`: Erosion followed by dilation
    - `"close"`: Dilation followed by erosion
    - `"erode"`: Erosion only
    - `"dilate"`: Dilation only
- **CSV Reference**: "Operações Morfológicas - Tamanho do elemento estruturante"
- **Implementation**: cv2.morphologyEx with elliptical kernel

### 18. **CompositeFilter**
- **Purpose**: Apply multiple filters in sequence
- **Parameters**: List of filter objects
- **Implementation**: Chains filters sequentially
- **Usage Examples**:
  - `"AHE40.0_CLAHE4.0"` → AHE then CLAHE
  - `"AHE40.0_CLAHE4.0_Gamma0.8"` → AHE → CLAHE → Gamma
  - `"AHE40.0_CLAHE4.0_Gamma0.8_MaxGreen2.0"` → Four-filter chain

## CSV Reference Mapping

| CSV Filter | Implemented Class | Parameters |
|------------|------------------|------------|
| Cropping/FOV | - | N/A (preprocessing step) |
| Resizing | - | N/A (preprocessing step) |
| Green channel | GreenChannelExtraction | None |
| Grayscale | GrayscaleConversion | None |
| Color Normalization (Ben Graham) | BenGrahamNormalization | None |
| Color Normalization (LAB) | LABNormalization | None |
| CLAHE | CLAHEFilter | clip_limit, tile_grid_size |
| Histogram Equalization | HistogramEqualization | None |
| Gamma Correction | GammaFilter | gamma |
| Illumination Correction | MultiScaleRetinex | scales |
| Gaussian Filter | GaussianBlur | sigma |
| Median Filter | MedianFilter | kernel_size |
| Canny Edge | CannyEdgeDetection | lower_threshold, upper_threshold |
| Frangi Vesselness | FrangiVesselness | scales, beta, c |
| Morphological Ops | MorphologicalOperations | kernel_size, operation |
| Otsu Thresholding | OtsuThresholding | None |
| Optic Disc Detection | - | N/A (not implemented as filter) |

## API Usage

### Single Filter
```python
from utils_preprocessing import get_filter

# Create filter with parameters
params = {"clip_limit": 4.0, "tile_grid_size": (8, 8)}
filter_obj = get_filter("CLAHE4.0", params)

# Apply to image
filtered_image = filter_obj.apply(image)

# Get configuration
config = filter_obj.get_config()
```

### Composite Filter
```python
# Automatic parsing of composite filter name
filter_obj = get_filter("AHE40.0_CLAHE4.0_Gamma0.8_MaxGreen2.0", {})

# Apply to image (applies filters in sequence)
filtered_image = filter_obj.apply(image)
```

### Parse Filter Name
```python
from utils_preprocessing import parse_filter_name

filter_name, params = parse_filter_name("AHE40.0_CLAHE4.0_Gamma0.8")
# Returns:
# filter_name = "AHE40.0_CLAHE4.0_Gamma0.8"
# params = {
#     "AHE": {"clip_limit": 40.0, "tile_grid_size": (8, 8)},
#     "CLAHE": {"clip_limit": 4.0, "tile_grid_size": (8, 8)},
#     "Gamma": {"gamma": 0.8}
# }
```

## Supported Filter Names (Case-Insensitive)

### Baseline
- `baseline`

### Single Filters
- `AHE40.0`, `AHE50.0`
- `CLAHE2.0`, `CLAHE4.0`
- `Gamma0.8`, `Gamma1.2`
- `MaxGreen2.0`, `MaxGreen3.0`, `MaxGreen4.0`

### Composite Filters (from filter_matrix_template.json)
- `AHE40.0_CLAHE4.0`
- `AHE40.0_CLAHE4.0_Gamma0.8`
- `AHE40.0_CLAHE4.0_Gamma0.8_MaxGreen2.0`
- `AHE40.0_CLAHE4.0_Gamma0.8_MaxGreen4.0`
- `AHE40.0_CLAHE4.0_Gamma1.2_MaxGreen2.0`
- `CLAHE4.0_Gamma0.8`
- `CLAHE4.0_Gamma1.2`

## Key Design Decisions

1. **RGB to LAB Conversion**: AHE and CLAHE filters convert to LAB colorspace and apply enhancements to the L (luminosity) channel only, preserving color information.

2. **Composite Filters**: Automatically parsed from underscore-separated names. Filters are applied in left-to-right order.

3. **Parameter Validation**: All filters include default parameters for robustness. Missing parameters fall back to sensible defaults.

4. **Compatibility**: Filters handle both RGB and grayscale images, with automatic conversion when needed.

5. **Non-Destructive**: All filters return new image objects; original images are not modified (`.copy()` used where appropriate).

## CSV Parameters Implementation

All parameters from `parametros-filtros-dr.csv` are incorporated:

- **CLAHE**: Clip limit range 2.0-5.5 supported; tile grid 8×8 or 32×32
- **Gamma**: Both brightening (0.8) and darkening (1.2) values supported
- **Green Channel**: Suppression factors 2.0-4.0 implemented
- **Gaussian**: Sigma range 5-10 pixels supported with formula σ = radius/30
- **Median**: Kernel sizes 3×3 and 5×5 supported
- **Canny**: Threshold ranges 6-10 (lower) and 32-60 (upper) as per CSV
- **Frangi**: Multi-scale (1-15), β=5, c=1-15 as per CSV
- **Morphological**: Kernel sizes 5×5, 11×11, 23×23 supported

## Testing

Run the test suite to verify all filters:
```bash
python3 test_preprocessing_filters.py
```

The test suite validates:
- ✓ All 19 filter classes load correctly
- ✓ Single filter application
- ✓ Composite filter chaining
- ✓ Filter name parsing
- ✓ Filters from filter_matrix_template.json
