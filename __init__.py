"""
Face Emotion Classification Module.

Real-time face emotion classification using TensorFlow Lite models
trained with Google's Teachable Machine.

Features:
- Real-time webcam emotion classification
- Batch dataset processing
- REST API server
- Single image classification
- Performance metrics tracking
"""

from .classifier import FaceClassifier
from .config import (
    PRODUCTION_CONFIG,
    REALTIME_CONFIG,
    BATCH_CONFIG,
    EmotionClassificationConfig
)

__version__ = "1.0.0"
__all__ = [
    "FaceClassifier",
    "PRODUCTION_CONFIG",
    "REALTIME_CONFIG",
    "BATCH_CONFIG",
    "EmotionClassificationConfig"
]
