# -*- coding: utf-8 -*-
"""
Utility module for image preprocessing filters.
Handles application of contrast enhancement and filtering techniques to retinal images.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class ImageFilter:
    """Base class for image filters."""
    
    def __init__(self, name: str, params: Dict[str, Any]):
        self.name = name
        self.params = params
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply filter to image. Should never be applied to masks."""
        raise NotImplementedError
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration dict for metadata logging."""
        return {
            "name": self.name,
            "params": self.params
        }


class IdentityFilter(ImageFilter):
    """Identity filter (no operation) - used for baseline."""
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Return image unchanged."""
        return image.copy()


class AHEFilter(ImageFilter):
    """Adaptive Histogram Equalization filter."""
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply AHE to image."""
        clip_limit = self.params.get("clip_limit", 40.0)
        tile_grid_size = self.params.get("tile_grid_size", (8, 8))
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert to LAB colorspace
            lab = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            l = clahe.apply(l)
            
            # Merge and convert back
            lab = cv2.merge([l, a, b])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            return result
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            return clahe.apply(image.astype(np.uint8))


class CLAHEFilter(ImageFilter):
    """Contrast Limited Adaptive Histogram Equalization filter."""
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE to image."""
        clip_limit = self.params.get("clip_limit", 4.0)
        tile_grid_size = self.params.get("tile_grid_size", (8, 8))
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            lab = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            l = clahe.apply(l)
            
            lab = cv2.merge([l, a, b])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            return result
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            return clahe.apply(image.astype(np.uint8))


class GammaFilter(ImageFilter):
    """Gamma correction filter."""
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply gamma correction to image."""
        gamma = self.params.get("gamma", 1.0)
        
        # Normalize to [0, 1]
        if image.dtype == np.uint8:
            image_norm = image.astype(np.float32) / 255.0
        else:
            image_norm = image.astype(np.float32) / 255.0 if image.max() > 1 else image.astype(np.float32)
        
        # Apply gamma correction
        corrected = np.power(image_norm, gamma)
        
        # Convert back to original range
        result = (corrected * 255).astype(np.uint8)
        return result


class MaxGreenFilter(ImageFilter):
    """Max green channel suppression filter."""
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Suppress green channel by scaling."""
        factor = self.params.get("max_green_factor", 1.0)
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = image.astype(np.float32)
            # Suppress green channel (typically channel 1 in RGB)
            image[:, :, 1] = image[:, :, 1] / factor
            return np.clip(image, 0, 255).astype(np.uint8)
        else:
            return image


class UnsharpMask(ImageFilter):
    """
    Unsharp masking (sharpening): subtracts a Gaussian-blurred ("unsharp")
    version of the image from the original to isolate high-frequency detail,
    then adds a scaled copy of that detail back to the original. Used in DR
    literature (e.g. bichannel-CNN referable-DR detectors) to accentuate
    fine structures such as microaneurysms and vessel edges before
    downstream feature extraction — the opposite operation of GaussianBlur.
    """

    def apply(self, image: np.ndarray) -> np.ndarray:
        amount = self.params.get("amount", 1.5)
        sigma = self.params.get("sigma", 5.0)

        image_f = image.astype(np.float32)
        blurred = cv2.GaussianBlur(image_f, (0, 0), sigma)
        sharpened = image_f + amount * (image_f - blurred)
        return np.clip(sharpened, 0, 255).astype(np.uint8)


class GreenChannelExtraction(ImageFilter):
    """Extract green channel from RGB image."""
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Extract green channel."""
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Extract green channel (index 1 in RGB)
            green = image[:, :, 1]
            return green
        else:
            # Already single channel
            return image


class GrayscaleConversion(ImageFilter):
    """Convert RGB to grayscale."""
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Convert to grayscale using standard weights."""
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Standard grayscale conversion
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            return gray
        else:
            # Already single channel
            return image


class HistogramEqualization(ImageFilter):
    """Global histogram equalization filter."""
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply global histogram equalization."""
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert to LAB, equalize L channel, convert back
            lab = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Equalize L channel
            l = cv2.equalizeHist(l)
            
            # Merge back
            lab = cv2.merge([l, a, b])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            return result
        else:
            return cv2.equalizeHist(image.astype(np.uint8))


class BenGrahamNormalization(ImageFilter):
    """
    Ben Graham local-average color subtraction, as popularized in the 2015
    Kaggle Diabetic Retinopathy Detection competition (1st place writeup).

    Formula: I' = clip(alpha*I + beta*G(sigma)*I + gamma), with the
    canonical values alpha=4, beta=-4, gamma=128 (i.e. `cv2.addWeighted`),
    where G(sigma) is a Gaussian blur estimating the local background
    illumination. Subtracting it removes illumination gradients/vignetting
    and boosts local contrast in one step.
    """

    def apply(self, image: np.ndarray) -> np.ndarray:
        sigma = self.params.get("sigma", 10.0)
        image_f = image.astype(np.float32)
        blurred = cv2.GaussianBlur(image_f, (0, 0), sigma)
        result = cv2.addWeighted(image_f, 4, blurred, -4, 128)
        return np.clip(result, 0, 255).astype(np.uint8)


class LABNormalization(ImageFilter):
    """LAB color space normalization."""
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize in LAB color space, particularly L (luminosity) channel.
        """
        image = image.astype(np.uint8)
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert to LAB
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Normalize L channel
            l_min = l.min()
            l_max = l.max()
            if l_max > l_min:
                l = ((l.astype(np.float32) - l_min) / (l_max - l_min) * 255).astype(np.uint8)
            
            # Merge back
            lab = cv2.merge([l, a, b])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            return result
        else:
            return image


class GaussianBlur(ImageFilter):
    """Gaussian blur filter for smoothing."""
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur."""
        sigma = self.params.get("sigma", 5.0)
        # Convert sigma to kernel size
        kernel_size = int(2 * np.ceil(3 * sigma) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        result = cv2.GaussianBlur(image.astype(np.uint8), (kernel_size, kernel_size), sigma)
        return result


class MedianFilter(ImageFilter):
    """Median filter for noise reduction."""
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply median filter."""
        kernel_size = self.params.get("kernel_size", 5)
        # Ensure odd kernel size
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        result = cv2.medianBlur(image.astype(np.uint8), kernel_size)
        return result


class MultiScaleRetinex(ImageFilter):
    """Multi-Scale Retinex for illumination correction."""
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Multi-Scale Retinex for illumination correction.
        Reference: ScienceDirect S2667099225000362
        """
        image = image.astype(np.float32)
        
        # Weights for multiple scales
        scales = [15, 80, 250]  # Multi-scale for different frequency components
        weights = [1/3, 1/3, 1/3]
        
        retinex = np.zeros_like(image)
        
        if len(image.shape) == 3:
            # Process each channel
            for c in range(image.shape[2]):
                channel = image[:, :, c]
                msr = np.zeros_like(channel)
                
                for scale, weight in zip(scales, weights):
                    # Gaussian blur with scale as sigma
                    gaussian = cv2.GaussianBlur(channel, (0, 0), scale)
                    # Retinex: log(input) - log(gaussian)
                    msr += weight * (np.log(channel + 1) - np.log(gaussian + 1))
                
                retinex[:, :, c] = msr
        else:
            msr = np.zeros_like(image)
            for scale, weight in zip(scales, weights):
                gaussian = cv2.GaussianBlur(image, (0, 0), scale)
                msr += weight * (np.log(image + 1) - np.log(gaussian + 1))
            retinex = msr
        
        # Normalize to [0, 255]
        retinex = retinex - retinex.min()
        if retinex.max() > 0:
            retinex = retinex / retinex.max() * 255.0
        
        return np.clip(retinex, 0, 255).astype(np.uint8)


class OtsuThresholding(ImageFilter):
    """Otsu automatic thresholding."""
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply Otsu thresholding."""
        if len(image.shape) == 3:
            # Convert to grayscale first
            image_gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image.astype(np.uint8)
        
        # Apply Otsu thresholding
        _, binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary


class CannyEdgeDetection(ImageFilter):
    """Canny edge detection filter."""
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply Canny edge detection."""
        lower_threshold = self.params.get("lower_threshold", 100)
        upper_threshold = self.params.get("upper_threshold", 200)
        
        if len(image.shape) == 3:
            # Convert to grayscale first
            image_gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image.astype(np.uint8)
        
        # Apply Canny edge detection
        edges = cv2.Canny(image_gray, lower_threshold, upper_threshold)
        
        return edges


class FrangiVesselness(ImageFilter):
    """Frangi vesselness filter for vessel detection."""
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Frangi vesselness filter for vessel detection.
        Reference: Merzoug & Yedjour, JISEM 2025
        """
        if len(image.shape) == 3:
            # Convert to grayscale
            image_gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image.astype(np.uint8)
        
        image_gray = image_gray.astype(np.float32) / 255.0
        
        scales = self.params.get("scales", [1, 2, 4, 8])
        beta = self.params.get("beta", 5)
        c = self.params.get("c", 15)
        
        frangi_output = np.zeros_like(image_gray)

        for scale in scales:
            # Compute Hessian matrix eigenvalues
            sigma = scale

            # Compute derivatives using Sobel
            a = cv2.Sobel(image_gray, cv2.CV_64F, 2, 0, ksize=5) * sigma  # Ixx
            d = cv2.Sobel(image_gray, cv2.CV_64F, 0, 2, ksize=5) * sigma  # Iyy
            b = cv2.Sobel(image_gray, cv2.CV_64F, 1, 1, ksize=5) * sigma  # Ixy

            # Closed-form eigenvalues of the symmetric 2x2 Hessian [[a,b],[b,d]]
            # at every pixel at once (equivalent to np.linalg.eigvalsh per
            # pixel, ascending order, but vectorized — see PRE_CAMPAIGN
            # investigation of job 800782/AHE40.0_Frangi: the original
            # per-pixel Python loop took ~66s/image at the 1280x1280 native
            # resolution used here, timing out the Script-1 SLURM job before
            # it could finish for any of the 72 Frangi-based filters).
            mean = (a + d) / 2.0
            tmp = np.sqrt(((a - d) / 2.0) ** 2 + b ** 2)
            lambda1, lambda2 = mean - tmp, mean + tmp

            Rb = lambda1**2 + lambda2**2
            S = np.sqrt(Rb)
            with np.errstate(divide='ignore', invalid='ignore'):
                Ra = np.where(lambda2 != 0, lambda1 / lambda2, 0.0)

            vesselness = (1 - np.exp(-Ra**2 / (2 * beta**2))) * \
                        (1 - np.exp(-Rb**2 / (2 * c**2))) * \
                        (1 - np.exp(-S**2 / (2 * c**2)))
            vesselness = np.where(lambda2 > 0, vesselness, 0.0)

            frangi_output = np.maximum(frangi_output, vesselness)

        # Normalize to [0, 255]
        frangi_output = (frangi_output - frangi_output.min()) / (frangi_output.max() - frangi_output.min() + 1e-7) * 255
        
        return frangi_output.astype(np.uint8)


class MorphologicalOperations(ImageFilter):
    """Morphological operations (opening, closing, etc.)."""
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply morphological operations."""
        kernel_size = self.params.get("kernel_size", 5)
        operation = self.params.get("operation", "open")  # open, close, erode, dilate
        
        # Create morphological kernel (elliptical)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        image = image.astype(np.uint8)
        
        if operation == "open":
            result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        elif operation == "close":
            result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        elif operation == "erode":
            result = cv2.erode(image, kernel, iterations=1)
        elif operation == "dilate":
            result = cv2.dilate(image, kernel, iterations=1)
        else:
            result = image
        
        return result


class CompositeFilter(ImageFilter):
    """Composite filter that applies multiple filters in sequence."""
    
    def __init__(self, name: str, filters: list):
        self.name = name
        self.filters = filters
        self.params = {f.name: f.params for f in filters}
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply all filters in sequence."""
        result = image.copy()
        for f in self.filters:
            result = f.apply(result)
        return result
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration dict for metadata logging."""
        return {
            "name": self.name,
            "filters": [f.get_config() for f in self.filters]
        }


def get_filter(filter_name: str, params: Dict[str, Any]) -> ImageFilter:
    """
    Factory function to get filter by name.
    Supports single and composite filters.
    """
    filter_name_lower = filter_name.lower()

    # Composite filters (contain underscore separators), or single filters of
    # a known parametrized type (AHE/CLAHE/Gamma/MaxGreen/Unsharp) are both
    # handled by _create_composite_filter(): it resolves prefixes via
    # startswith() (so "CLAHE4.0" never gets mistaken for "AHE...") and
    # re-derives the parameter values directly from the name string, rather
    # than relying on a possibly mismatched `params` dict shape.
    known_prefixes = ("ahe", "clahe", "gamma", "maxgreen", "unsharp")
    if "_" in filter_name or filter_name_lower.startswith(known_prefixes):
        return _create_composite_filter(filter_name, params)

    # Single filters
    if filter_name_lower == "baseline":
        # Baseline: identity filter (no operation)
        return IdentityFilter(filter_name, params)
    elif "green" in filter_name_lower:
        return GreenChannelExtraction(filter_name, params)
    elif "grayscale" in filter_name_lower or "gray" in filter_name_lower:
        return GrayscaleConversion(filter_name, params)
    elif "histogram" in filter_name_lower or "equaliz" in filter_name_lower:
        return HistogramEqualization(filter_name, params)
    elif "bengra" in filter_name_lower or "ben_gra" in filter_name_lower:
        return BenGrahamNormalization(filter_name, params)
    elif "labnorm" in filter_name_lower or "lab" in filter_name_lower:
        return LABNormalization(filter_name, params)
    elif "gaussian" in filter_name_lower or "blur" in filter_name_lower:
        return GaussianBlur(filter_name, params)
    elif "median" in filter_name_lower:
        return MedianFilter(filter_name, params)
    elif "retinex" in filter_name_lower or "illumin" in filter_name_lower:
        return MultiScaleRetinex(filter_name, params)
    elif "otsu" in filter_name_lower or "threshold" in filter_name_lower:
        return OtsuThresholding(filter_name, params)
    elif "canny" in filter_name_lower or "edge" in filter_name_lower:
        return CannyEdgeDetection(filter_name, params)
    elif "frangi" in filter_name_lower or "vessel" in filter_name_lower:
        return FrangiVesselness(filter_name, params)
    elif "morpho" in filter_name_lower:
        return MorphologicalOperations(filter_name, params)
    else:
        raise ValueError(f"Unknown filter: {filter_name}")


def _create_composite_filter(filter_name: str, params: Dict[str, Any]) -> CompositeFilter:
    """
    Create a composite filter from a filter name like 'AHE40.0_CLAHE4.0_Gamma0.8_MaxGreen2.0'
    or 'BenGraham_CLAHE4.0_Frangi'.
    """
    parts = filter_name.split("_")
    filters = []
    
    # Track which parts have been used to avoid duplicates
    used_indices = set()
    
    for i, part in enumerate(parts):
        if i in used_indices:
            continue
        
        part_lower = part.lower()
        
        # --- PARAMETRIZED FILTERS ---
        if part_lower == "baseline":
            filters.append(IdentityFilter("baseline", {}))
            used_indices.add(i)
        
        elif part_lower.startswith("ahe"):
            try:
                clip_limit = float(part[3:])
                params_ahe = {"clip_limit": clip_limit, "tile_grid_size": (8, 8)}
                filters.append(AHEFilter(part, params_ahe))
                used_indices.add(i)
            except ValueError:
                pass
        
        elif part_lower.startswith("clahe"):
            try:
                clip_limit = float(part[5:])
                params_clahe = {"clip_limit": clip_limit, "tile_grid_size": (8, 8)}
                filters.append(CLAHEFilter(part, params_clahe))
                used_indices.add(i)
            except ValueError:
                pass
        
        elif part_lower.startswith("gamma"):
            try:
                gamma = float(part[5:])
                params_gamma = {"gamma": gamma}
                filters.append(GammaFilter(part, params_gamma))
                used_indices.add(i)
            except ValueError:
                pass
        
        elif part_lower.startswith("maxgreen"):
            try:
                factor = float(part[8:])
                params_mg = {"max_green_factor": factor}
                filters.append(MaxGreenFilter(part, params_mg))
                used_indices.add(i)
            except ValueError:
                pass

        elif part_lower.startswith("unsharp"):
            try:
                amount = float(part[7:])
                params_unsharp = {"amount": amount, "sigma": 5.0}
                filters.append(UnsharpMask(part, params_unsharp))
                used_indices.add(i)
            except ValueError:
                pass
                
        # --- NON-PARAMETRIZED FILTERS FALLBACK (THE FIX) ---
        elif "green" in part_lower and not part_lower.startswith("maxgreen"):
            filters.append(GreenChannelExtraction(part, {}))
            used_indices.add(i)
        elif "grayscale" in part_lower or "gray" in part_lower:
            filters.append(GrayscaleConversion(part, {}))
            used_indices.add(i)
        elif "histogram" in part_lower or "equaliz" in part_lower:
            filters.append(HistogramEqualization(part, {}))
            used_indices.add(i)
        elif "bengra" in part_lower or "ben_gra" in part_lower:
            filters.append(BenGrahamNormalization(part, {}))
            used_indices.add(i)
        elif "labnorm" in part_lower or "lab" in part_lower:
            filters.append(LABNormalization(part, {}))
            used_indices.add(i)
        elif "gaussian" in part_lower or "blur" in part_lower:
            filters.append(GaussianBlur(part, {}))
            used_indices.add(i)
        elif "median" in part_lower:
            filters.append(MedianFilter(part, {}))
            used_indices.add(i)
        elif "retinex" in part_lower or "illumin" in part_lower:
            filters.append(MultiScaleRetinex(part, {}))
            used_indices.add(i)
        elif "otsu" in part_lower or "threshold" in part_lower:
            filters.append(OtsuThresholding(part, {}))
            used_indices.add(i)
        elif "canny" in part_lower or "edge" in part_lower:
            filters.append(CannyEdgeDetection(part, {}))
            used_indices.add(i)
        elif "frangi" in part_lower or "vessel" in part_lower:
            filters.append(FrangiVesselness(part, {}))
            used_indices.add(i)
        elif "morpho" in part_lower:
            filters.append(MorphologicalOperations(part, {}))
            used_indices.add(i)
        else:
            logger.warning(f"Could not parse or unrecognized composite part: {part}")

    if not filters:
        # If no filters were parsed, return identity
        filters.append(IdentityFilter("baseline", {}))
    
    return CompositeFilter(filter_name, filters)


def parse_filter_name(filter_str: str) -> Tuple[str, Dict[str, Any]]:
    """
    Parse filter string like 'AHE40.0_CLAHE4.0_Gamma0.8_MaxGreen2.0' into name and params.
    
    Returns:
        Tuple of (filter_name, params_dict)
    
    Examples:
        'baseline' -> ('baseline', {})
        'AHE40.0' -> ('AHE40.0', {'clip_limit': 40.0, 'tile_grid_size': (8, 8)})
        'CLAHE4.0' -> ('CLAHE4.0', {'clip_limit': 4.0, 'tile_grid_size': (8, 8)})
        'Gamma0.8' -> ('Gamma0.8', {'gamma': 0.8})
        'MaxGreen2.0' -> ('MaxGreen2.0', {'max_green_factor': 2.0})
        'AHE40.0_CLAHE4.0_Gamma0.8_MaxGreen2.0' -> composite params
    """
    params = {}
    parts = filter_str.split("_")
    
    # List of known valid keywords that do not extract numeric parameters directly from string
    valid_non_param_keywords = [
        "green", "gray", "histogram", "equaliz", "bengra", "labnorm", 
        "gaussian", "blur", "median", "retinex", "illumin", "otsu", 
        "threshold", "canny", "edge", "frangi", "vessel", "morpho"
    ]
    
    for part in parts:
        part_lower = part.lower()
        
        if part_lower == "baseline":
            # Baseline has no parameters
            continue
        
        elif part_lower.startswith("ahe"):
            try:
                clip_limit = float(part[3:])
                if "AHE" not in params:
                    params["AHE"] = {"clip_limit": clip_limit, "tile_grid_size": (8, 8)}
            except ValueError:
                logger.warning(f"Could not parse AHE parameters from: {part}")
        
        elif part_lower.startswith("clahe"):
            try:
                clip_limit = float(part[5:])
                if "CLAHE" not in params:
                    params["CLAHE"] = {"clip_limit": clip_limit, "tile_grid_size": (8, 8)}
            except ValueError:
                logger.warning(f"Could not parse CLAHE parameters from: {part}")
        
        elif part_lower.startswith("gamma"):
            try:
                gamma = float(part[5:])
                if "Gamma" not in params:
                    params["Gamma"] = {"gamma": gamma}
            except ValueError:
                logger.warning(f"Could not parse Gamma parameters from: {part}")
        
        elif part_lower.startswith("maxgreen"):
            try:
                factor = float(part[8:])
                if "MaxGreen" not in params:
                    params["MaxGreen"] = {"max_green_factor": factor}
            except ValueError:
                logger.warning(f"Could not parse MaxGreen parameters from: {part}")

        elif part_lower.startswith("unsharp"):
            try:
                amount = float(part[7:])
                if "Unsharp" not in params:
                    params["Unsharp"] = {"amount": amount, "sigma": 5.0}
            except ValueError:
                logger.warning(f"Could not parse Unsharp parameters from: {part}")
                
        # --- FIX: Acknowledge valid non-parameterized filters to avoid false warnings ---
        elif any(kw in part_lower for kw in valid_non_param_keywords):
            # These filters exist but don't parse dynamic numbers from their names
            params[part] = {}
            continue
            
        else:
            logger.warning(f"Unrecognized filter part during parsing: {part}")

    return filter_str, params