"""
FaceXHuBERT: Text-less Speech-driven E(X)pressive 3D Facial Animation Synthesis

This package provides tools for generating expressive 3D facial animations from speech audio
using self-supervised speech representation learning with HuBERT.

Main components:
- FaceXHuBERT: The main model class
- predict: Prediction and inference functionality
- render_result: Rendering utilities for visualization
- hubert: Custom HuBERT implementation
"""

__version__ = "1.0.0"
__author__ = "Kazi Injamamul Haque, Zerrin Yumak"
__license__ = "CC-BY-NC-4.0"

from .faceXhubert import FaceXHuBERT

__all__ = [
    "FaceXHuBERT",
]
