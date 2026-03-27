"""
REST API server for face emotion classification.

Provides HTTP endpoints for:
- Single image classification
- Batch processing
- Health checks
- Performance metrics
"""

import logging
import time
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from .classifier import FaceClassifier
from .config import EmotionClassificationConfig

logger = logging.getLogger(__name__)


def create_app(classifier: FaceClassifier) -> Flask:
    """
    Create Flask application with API routes.
    
    Args:
        classifier: FaceClassifier instance
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    
    # Metrics tracking
    metrics = {
        'requests_total': 0,
        'requests_success': 0,
        'requests_failed': 0,
        'inference_times': [],
    }
    
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        return jsonify({
            'status': 'healthy',
            'model_loaded': classifier.interpreter is not None,
            'timestamp': time.time(),
        })
    
    @app.route('/api/classify', methods=['POST'])
    def classify_image():
        """
        Classify a single image.
        
        Expects:
        - multipart/form-data with 'image' file field
        OR
        - JSON with 'image_path' field
        
        Returns:
        - Classification result with emotion and confidence
        """
        start_time = time.time()
        metrics['requests_total'] += 1
        
        try:
            # Handle file upload
            if 'image' in request.files:
                file = request.files['image']
                if file.filename == '':
                    return jsonify({'error': 'No file selected'}), 400
                
                # Save temporarily
                import tempfile
                import cv2
                import numpy as np
                
                filename = secure_filename(file.filename)
                
                # Read file into numpy array
                nparr = np.frombuffer(file.read(), np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    return jsonify({'error': 'Invalid image file'}), 400
                    
            elif request.is_json and 'image_path' in request.json:
                image_path = request.json['image_path']
                image = None  # Will be loaded by classifier
            else:
                return jsonify({'error': 'No image provided'}), 400
            
            # Classify
            if image is not None:
                result = classifier.classify_image(image)
            else:
                result = classifier.classify_image_file(image_path)
            
            inference_time = time.time() - start_time
            metrics['inference_times'].append(inference_time)
            
            if len(metrics['inference_times']) > 100:
                metrics['inference_times'].pop(0)
            
            if result:
                metrics['requests_success'] += 1
                
                return jsonify({
                    'success': True,
                    'emotion': result.emotion,
                    'confidence': result.confidence,
                    'all_probabilities': result.all_probabilities,
                    'inference_time_ms': inference_time * 1000,
                })
            else:
                metrics['requests_failed'] += 1
                return jsonify({
                    'success': False,
                    'error': 'Classification failed',
                }), 500
                
        except Exception as e:
            logger.error(f"Classification error: {e}", exc_info=True)
            metrics['requests_failed'] += 1
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/metrics', methods=['GET'])
    def get_metrics():
        """Get performance metrics."""
        avg_inference_time = (
            sum(metrics['inference_times']) / len(metrics['inference_times'])
            if metrics['inference_times'] else 0
        )
        
        return jsonify({
            'requests_total': metrics['requests_total'],
            'requests_success': metrics['requests_success'],
            'requests_failed': metrics['requests_failed'],
            'success_rate': (
                metrics['requests_success'] / metrics['requests_total']
                if metrics['requests_total'] > 0 else 0
            ),
            'avg_inference_time_ms': avg_inference_time * 1000,
        })
    
    @app.route('/api/config', methods=['GET'])
    def get_config():
        """Get current configuration."""
        return jsonify({
            'model_path': classifier.config.model_path,
            'labels_path': classifier.config.labels_path,
            'confidence_threshold': classifier.config.confidence_threshold,
            'input_size': classifier.config.input_size,
            'num_threads': classifier.config.num_threads,
        })
    
    @app.route('/', methods=['GET'])
    def index():
        """API documentation endpoint."""
        return jsonify({
            'name': 'Face Emotion Classification API',
            'version': '1.0.0',
            'endpoints': [
                {'path': '/', 'method': 'GET', 'description': 'API info'},
                {'path': '/api/health', 'method': 'GET', 'description': 'Health check'},
                {'path': '/api/classify', 'method': 'POST', 'description': 'Classify image'},
                {'path': '/api/metrics', 'method': 'GET', 'description': 'Performance metrics'},
                {'path': '/api/config', 'method': 'GET', 'description': 'Configuration'},
            ],
        })
    
    return app
