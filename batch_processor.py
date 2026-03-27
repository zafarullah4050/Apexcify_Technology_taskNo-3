"""
Batch processing for face emotion classification.

Process large datasets of images with statistics and progress tracking.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from tqdm import tqdm

from .classifier import FaceClassifier, ClassificationResult

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Batch image classifier for datasets."""
    
    def __init__(self, classifier: FaceClassifier):
        """
        Initialize batch processor.
        
        Args:
            classifier: FaceClassifier instance
        """
        self.classifier = classifier
        
    def process_directory(
        self,
        directory: str,
        output_path: Optional[str] = None,
        recursive: bool = True,
    ) -> Dict:
        """
        Process all images in a directory.
        
        Args:
            directory: Directory containing images
            output_path: Optional path to save results JSON
            recursive: Whether to search subdirectories
            
        Returns:
            Dictionary with results and statistics
        """
        dir_path = Path(directory)
        
        if not dir_path.exists():
            logger.error(f"Directory not found: {directory}")
            return {"error": "Directory not found"}
        
        # Find all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
        image_files = []
        
        if recursive:
            for ext in image_extensions:
                image_files.extend(dir_path.rglob(ext))
        else:
            for ext in image_extensions:
                image_files.extend(dir_path.glob(ext))
        
        if not image_files:
            logger.warning(f"No images found in {directory}")
            return {"error": "No images found", "count": 0}
        
        logger.info(f"Found {len(image_files)} images in {directory}")
        
        # Process images
        results = []
        statistics = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "emotions": {},
        }
        
        # Use tqdm for progress
        for image_path in tqdm(image_files, desc="Processing images"):
            result = self.classifier.classify_image_file(str(image_path))
            
            statistics["total"] += 1
            
            if result:
                statistics["success"] += 1
                
                # Count emotions
                emotion = result.emotion
                if emotion not in statistics["emotions"]:
                    statistics["emotions"][emotion] = {
                        "count": 0,
                        "confidences": [],
                    }
                
                statistics["emotions"][emotion]["count"] += 1
                statistics["emotions"][emotion]["confidences"].append(result.confidence)
                
                results.append({
                    "path": str(image_path),
                    "emotion": result.emotion,
                    "confidence": result.confidence,
                    "all_probabilities": result.all_probabilities,
                })
            else:
                statistics["failed"] += 1
                results.append({
                    "path": str(image_path),
                    "error": "Classification failed",
                })
        
        # Calculate average confidences per emotion
        for emotion in statistics["emotions"].values():
            if emotion["confidences"]:
                emotion["avg_confidence"] = sum(emotion["confidences"]) / len(emotion["confidences"])
                del emotion["confidences"]  # Remove raw data to save space
        
        # Prepare final report
        report = {
            "statistics": statistics,
            "results": results,
        }
        
        # Save to file if requested
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Results saved to: {output_path}")
        
        # Print summary
        self._print_summary(statistics)
        
        return report
    
    def _print_summary(self, statistics: Dict) -> None:
        """Print processing summary."""
        print("\n" + "="*60)
        print("BATCH PROCESSING SUMMARY")
        print("="*60)
        print(f"Total images:     {statistics['total']}")
        print(f"Successful:       {statistics['success']}")
        print(f"Failed:           {statistics['failed']}")
        
        if statistics['success'] > 0:
            success_rate = (statistics['success'] / statistics['total']) * 100
            print(f"Success rate:     {success_rate:.1f}%")
        
        print("\nEmotion Distribution:")
        for emotion, data in statistics['emotions'].items():
            percentage = (data['count'] / statistics['success']) * 100 if statistics['success'] > 0 else 0
            avg_conf = data.get('avg_confidence', 0) * 100
            print(f"  {emotion:20s}: {data['count']:4d} ({percentage:5.1f}%) - Avg confidence: {avg_conf:.1f}%")
        
        print("="*60 + "\n")
