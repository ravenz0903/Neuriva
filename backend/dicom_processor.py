"""
dicom_processor.py — Clinical Radiologic Image Ingestion
=========================================================
Handles DICOM (.dcm) files with true Hounsfield Unit calibration
and stroke windowing, plus standard raster image fallback (PNG/JPG).
"""

import io
import numpy as np
from PIL import Image

# Stroke CT windowing parameters
STROKE_WINDOW_WIDTH = 80   # WW
STROKE_WINDOW_LEVEL = 40   # WL
# Derived bounds: [WL - WW/2, WL + WW/2] = [0, 80] HU


def read_dicom(file_bytes: bytes) -> np.ndarray:
    """
    Read a DICOM file from raw bytes, apply Hounsfield calibration
    and stroke windowing. Returns 8-bit grayscale numpy array.
    """
    import pydicom

    ds = pydicom.dcmread(io.BytesIO(file_bytes))
    pixel_array = ds.pixel_array.astype(np.float64)

    # Apply Hounsfield calibration: HU = pixel_value * slope + intercept
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu_array = pixel_array * slope + intercept

    # Apply stroke window
    return apply_stroke_window(hu_array)


def read_raster_image(file_bytes: bytes) -> np.ndarray:
    """
    Read a standard raster image (PNG, JPG, BMP, TIFF) from raw bytes.
    Converts to 8-bit grayscale numpy array.
    """
    img = Image.open(io.BytesIO(file_bytes)).convert("L")
    return np.array(img, dtype=np.uint8)


def apply_stroke_window(hu_array: np.ndarray) -> np.ndarray:
    """
    Apply stroke CT window (WW=80, WL=40) to Hounsfield Unit array.
    Maps [WL - WW/2, WL + WW/2] = [0, 80] HU -> [0, 255] uint8.
    """
    lower = STROKE_WINDOW_LEVEL - STROKE_WINDOW_WIDTH / 2  # 0 HU
    upper = STROKE_WINDOW_LEVEL + STROKE_WINDOW_WIDTH / 2  # 80 HU

    windowed = np.clip(hu_array, lower, upper)
    # Normalize to 0-255
    windowed = ((windowed - lower) / (upper - lower) * 255.0)
    return windowed.astype(np.uint8)


def process_upload(file_bytes: bytes, filename: str) -> np.ndarray:
    """
    Unified entry point. Dispatches to DICOM or raster reader based
    on file extension.

    Returns:
        np.ndarray: 8-bit grayscale image (H, W), ready for model inference.
    """
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if ext == "dcm":
        return read_dicom(file_bytes)
    elif ext in ("png", "jpg", "jpeg", "bmp", "tiff", "tif"):
        return read_raster_image(file_bytes)
    else:
        # Try DICOM first (some DICOM files have no extension), fall back to raster
        try:
            return read_dicom(file_bytes)
        except Exception:
            return read_raster_image(file_bytes)
