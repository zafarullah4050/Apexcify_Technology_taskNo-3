"""
Real-time webcam interface for face emotion classification.

Captures video from webcam and performs real-time emotion classification
with visual feedback.
"""

import logging
import time
from typing import Optional

import cv2
import numpy as np

from .classifier import FaceClassifier, ClassificationResult
from .config import EmotionClassificationConfig

logger = logging.getLogger(__name__)


class WebcamClassifier:
    """Real-time webcam emotion classifier."""
    
    def __init__(self, classifier: FaceClassifier, webcam_id: int = 0):
        """
        Initialize webcam classifier.
        
        Args:
            classifier: FaceClassifier instance
            webcam_id: Webcam device ID (default: 0)
        """
        self.classifier = classifier
        self.webcam_id = webcam_id
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.is_paused = False
        
        # Performance tracking
        self.frame_count = 0
        self.fps_history = []
        self.start_time: Optional[float] = None
        
        # Current result
        self.current_result: Optional[ClassificationResult] = None
        self.result_timestamp: Optional[float] = None
        
    def initialize_webcam(self) -> bool:
        """
        Initialize webcam capture.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Initializing webcam (device {self.webcam_id})...")
            self.cap = cv2.VideoCapture(self.webcam_id)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open webcam device {self.webcam_id}")
                return False
            
            # Get camera properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            
            logger.info(f"Webcam opened: {width}x{height}@{fps:.1f}fps")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize webcam: {e}", exc_info=True)
            return False
    
    def _draw_result(self, frame: np.ndarray, result: ClassificationResult) -> np.ndarray:
        """
        Draw classification result on frame.
        
        Args:
            frame: Input frame
            result: Classification result
            
        Returns:
            Frame with annotations
        """
        # Get colors based on emotion
        emotion_colors = {
            "Happy Face": (0, 255, 0),      # Green
            "Sad Face": (255, 0, 0),        # Blue
            "Neutral": (200, 200, 200),     # Gray
        }
        
        color = emotion_colors.get(result.emotion, (0, 255, 255))  # Yellow default
        
        # Draw background rectangle
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 60), color, -1)
        
        # Draw text
        text = f"{result.emotion}: {result.confidence:.1%}"
        cv2.putText(
            frame, text, (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.classifier.config.font_scale,
            (255, 255, 255),
            self.classifier.config.thickness,
        )
        
        # Draw all probabilities
        y_offset = 70
        for emotion, prob in sorted(result.all_probabilities.items(), 
                                   key=lambda x: x[1], reverse=True):
            text = f"{emotion}: {prob:.1%}"
            cv2.putText(
                frame, text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
            y_offset += 25
        
        return frame
    
    def _draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Draw FPS counter on frame."""
        h, w = frame.shape[:2]
        text = f"FPS: {fps:.1f}"
        cv2.putText(
            frame, text, (w - 150, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        return frame
    
    def _update_fps(self) -> float:
        """Update and calculate FPS."""
        current_time = time.time()
        
        if self.start_time is None:
            self.start_time = current_time
            return 0.0
        
        elapsed = current_time - self.start_time
        self.frame_count += 1
        
        if elapsed > 0:
            current_fps = self.frame_count / elapsed
            self.fps_history.append(current_fps)
            
            # Keep only last 30 FPS measurements
            if len(self.fps_history) > 30:
                self.fps_history.pop(0)
            
            return sum(self.fps_history) / len(self.fps_history)
        
        return 0.0
    
    def run(self, save_video: bool = False) -> None:
        """
        Run real-time classification.
        
        Args:
            save_video: Whether to save output video
        """
        if not self.cap or not self.cap.isOpened():
            if not self.initialize_webcam():
                return
        
        logger.info("Starting real-time emotion classification...")
        logger.info("Controls: 'q' quit, 'p' pause, 's' save frame")
        
        # Video writer setup
        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            
            output_path = f"emotion_output_{int(time.time())}.avi"
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            logger.info(f"Saving output video to: {output_path}")
        
        self.is_running = True
        self.frame_count = 0
        self.fps_history = []
        self.start_time = time.time()
        
        inference_interval = 0.1  # Run inference every 100ms
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.warning("Failed to read frame")
                    continue
                
                self.frame_count += 1
                current_time = time.time()
                
                # Update FPS
                avg_fps = self._update_fps()
                
                # Run inference periodically
                if not self.is_paused and (current_time - (self.result_timestamp or 0)) >= inference_interval:
                    result = self.classifier.classify_image(frame)
                    if result:
                        self.current_result = result
                        self.result_timestamp = current_time
                
                # Draw results
                display_frame = frame.copy()
                
                if self.current_result and not self.is_paused:
                    display_frame = self._draw_result(display_frame, self.current_result)
                
                if self.classifier.config.show_fps:
                    display_frame = self._draw_fps(display_frame, avg_fps)
                
                # Save to video
                if video_writer:
                    video_writer.write(frame)
                
                # Display
                cv2.imshow("Face Emotion Classifier", display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    logger.info("Quit requested")
                    break
                elif key == ord('p'):
                    self.is_paused = not self.is_paused
                    logger.info(f"{'Paused' if self.is_paused else 'Resumed'}")
                elif key == ord('s'):
                    filename = f"frame_{self.frame_count:06d}.jpg"
                    cv2.imwrite(filename, frame)
                    logger.info(f"Saved frame: {filename}")
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            # Cleanup
            if video_writer:
                video_writer.release()
                logger.info(f"Video saved successfully")
            
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up...")
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        logger.info(f"Classification complete. Processed {self.frame_count} frames.")
        logger.info(f"Average FPS: {sum(self.fps_history)/len(self.fps_history) if self.fps_history else 0:.1f}")
