"""
Configuration management for face emotion classification.

Provides predefined configurations for different use cases:
- PRODUCTION_CONFIG: Best accuracy, suitable for deployment
- REALTIME_CONFIG: Low latency for real-time applications
- BATCH_CONFIG: Optimized for processing large datasets
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class EmotionClassificationConfig:
    """Configuration for emotion classification pipeline."""
    
    # Model settings
    model_path: str = "./face_emotion_model/model.tflite"
    labels_path: str = "./face_emotion_model/labels.txt"
    
    # Inference settings
    confidence_threshold: float = 0.75
    input_size: Tuple[int, int] = (224, 224)
    num_threads: int = 4
    
    # Performance settings
    batch_size: int = 1
    enable_tflite_delegates: bool = True
    
    # Display settings
    show_confidence: bool = True
    show_fps: bool = True
    font_scale: float = 1.0
    thickness: int = 2


# Production configuration - best accuracy
PRODUCTION_CONFIG = EmotionClassificationConfig(
    confidence_threshold=0.75,
    num_threads=4,
    batch_size=1,
    enable_tflite_delegates=True,
    show_confidence=True,
    show_fps=True,
)

# Real-time configuration - low latency
REALTIME_CONFIG = EmotionClassificationConfig(
    confidence_threshold=0.70,
    num_threads=2,
    batch_size=1,
    enable_tflite_delegates=False,  # Lower overhead
    show_confidence=True,
    show_fps=True,
)

# Batch processing configuration - large datasets
BATCH_CONFIG = EmotionClassificationConfig(
    confidence_threshold=0.80,
    num_threads=8,
    batch_size=32,
    enable_tflite_delegates=True,
    show_confidence=True,
    show_fps=False,
)
