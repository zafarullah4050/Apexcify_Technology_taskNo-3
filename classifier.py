"""
Face emotion classifier using TensorFlow Lite.

Provides efficient inference with TFLite models exported from
Google's Teachable Machine.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf

from .config import EmotionClassificationConfig

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of a classification."""
    
    emotion: str
    confidence: float
    all_probabilities: dict
    
    def __str__(self) -> str:
        return f"{self.emotion} ({self.confidence:.1%})"


class FaceClassifier:
    """
    Face emotion classifier using TensorFlow Lite.
    
    Supports:
    - Single image classification
    - Batch processing
    - Real-time webcam classification
    """
    
    def __init__(self, config: EmotionClassificationConfig):
        """
        Initialize the classifier.
        
        Args:
            config: Configuration for the classifier
        """
        self.config = config
        self.interpreter: Optional[tf.lite.Interpreter] = None
        self.labels: List[str] = []
        self.input_details: List[dict] = []
        self.output_details: List[dict] = []
        
    def load_model(self) -> bool:
        """
        Load the TFLite model and labels.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Verify model file exists
            model_path = Path(self.config.model_path)
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Load labels
            labels_path = Path(self.config.labels_path)
            if labels_path.exists():
                with open(labels_path, 'r') as f:
                    self.labels = [line.strip().split(' ', 1)[1] if ' ' in line else line.strip() 
                                   for line in f.readlines()]
                logger.info(f"Loaded {len(self.labels)} class labels")
            else:
                logger.warning(f"Labels file not found: {labels_path}")
                self.labels = [f"Class {i}" for i in range(3)]
            
            # Initialize TFLite interpreter
            self.interpreter = tf.lite.Interpreter(
                model_path=str(model_path),
                num_threads=self.config.num_threads,
            )
            
            # Allocate tensors
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            logger.info(f"Model loaded successfully: {model_path}")
            logger.info(f"Input shape: {self.input_details[0]['shape']}")
            logger.info(f"Output shape: {self.output_details[0]['shape']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            return False
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Preprocessed image tensor
        """
        # Resize to model input size
        resized = cv2.resize(image, self.config.input_size)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)
        
        return batched
    
    def classify_image(self, image: np.ndarray) -> Optional[ClassificationResult]:
        """
        Classify a single image.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Classification result or None if failed
        """
        if self.interpreter is None:
            logger.error("Model not loaded. Call load_model() first.")
            return None
        
        try:
            # Preprocess
            input_data = self._preprocess_image(image)
            
            # Set input tensor
            self.interpreter.set_tensor(
                self.input_details[0]['index'], 
                input_data
            )
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            output_data = self.interpreter.get_tensor(
                self.output_details[0]['index']
            )
            
            # Get probabilities
            probabilities = output_data[0]
            
            # Get prediction
            predicted_index = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_index])
            
            # Get label
            if 0 <= predicted_index < len(self.labels):
                emotion = self.labels[predicted_index]
            else:
                emotion = f"Class {predicted_index}"
            
            # Create result
            all_probs = {
                label: float(probabilities[i]) if i < len(probabilities) else 0.0
                for i, label in enumerate(self.labels)
            }
            
            result = ClassificationResult(
                emotion=emotion,
                confidence=confidence,
                all_probabilities=all_probs
            )
            
            logger.debug(f"Classification: {emotion} ({confidence:.1%})")
            
            return result
            
        except Exception as e:
            logger.error(f"Classification failed: {e}", exc_info=True)
            return None
    
    def classify_image_file(self, image_path: str) -> Optional[ClassificationResult]:
        """
        Classify an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Classification result or None if failed
        """
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            
            return self.classify_image(image)
            
        except Exception as e:
            logger.error(f"Failed to classify image file: {e}", exc_info=True)
            return None
    
    def batch_classify(self, image_paths: List[str]) -> List[Tuple[str, Optional[ClassificationResult]]]:
        """
        Classify multiple images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of (path, result) tuples
        """
        results = []
        
        for path in image_paths:
            result = self.classify_image_file(path)
            results.append((path, result))
            
            # Progress logging
            if len(results) % 10 == 0:
                logger.info(f"Processed {len(results)}/{len(image_paths)} images")
        
        logger.info(f"Batch classification complete: {len(results)} images")
        return results
