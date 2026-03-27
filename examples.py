"""
Example usage of the face emotion classification module.

Demonstrates:
- Single image classification
- Real-time webcam classification
- Batch processing
- REST API server
"""

import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def example_single_image():
    """Example 1: Classify a single image."""
    from .classifier import FaceClassifier
    from .config import PRODUCTION_CONFIG
    
    print("\n" + "="*60)
    print("EXAMPLE 1: Single Image Classification")
    print("="*60)
    
    # Initialize classifier
    classifier = FaceClassifier(PRODUCTION_CONFIG)
    
    if not classifier.load_model():
        print("Failed to load model!")
        return
    
    # Classify an image
    image_path = "./images/simple.jpg"
    if Path(image_path).exists():
        result = classifier.classify_image_file(image_path)
        
        if result:
            print(f"\nImage: {image_path}")
            print(f"Emotion: {result.emotion}")
            print(f"Confidence: {result.confidence:.1%}")
            print("\nAll probabilities:")
            for emotion, prob in result.all_probabilities.items():
                print(f"  {emotion}: {prob:.1%}")
        else:
            print("Classification failed!")
    else:
        print(f"Image not found: {image_path}")
        print("Skipping this example")


def example_webcam():
    """Example 2: Real-time webcam classification."""
    from .classifier import FaceClassifier
    from .webcam_interface import WebcamClassifier
    from .config import REALTIME_CONFIG
    
    print("\n" + "="*60)
    print("EXAMPLE 2: Real-time Webcam Classification")
    print("="*60)
    print("\nInitializing webcam...")
    print("Controls: 'q' quit, 'p' pause, 's' save frame")
    
    # Initialize classifier
    classifier = FaceClassifier(REALTIME_CONFIG)
    
    if not classifier.load_model():
        print("Failed to load model!")
        return
    
    # Initialize webcam
    webcam = WebcamClassifier(classifier, webcam_id=0)
    
    if not webcam.initialize_webcam():
        print("Failed to initialize webcam!")
        return
    
    # Run (optionally save video)
    webcam.run(save_video=False)


def example_batch_processing():
    """Example 3: Batch process a directory of images."""
    from .classifier import FaceClassifier
    from .batch_processor import BatchProcessor
    from .config import BATCH_CONFIG
    
    print("\n" + "="*60)
    print("EXAMPLE 3: Batch Processing")
    print("="*60)
    
    # Initialize classifier
    classifier = FaceClassifier(BATCH_CONFIG)
    
    if not classifier.load_model():
        print("Failed to load model!")
        return
    
    # Initialize batch processor
    processor = BatchProcessor(classifier)
    
    # Process images directory
    images_dir = "./images"
    output_file = "./batch_results.json"
    
    if Path(images_dir).exists():
        results = processor.process_directory(
            directory=images_dir,
            output_path=output_file,
            recursive=False,
        )
        
        if "error" not in results:
            print(f"\nBatch processing complete!")
            print(f"Results saved to: {output_file}")
    else:
        print(f"Directory not found: {images_dir}")
        print("Skipping this example")


def example_api_server():
    """Example 4: Start REST API server."""
    from .classifier import FaceClassifier
    from .api_server import create_app
    from .config import PRODUCTION_CONFIG
    
    print("\n" + "="*60)
    print("EXAMPLE 4: REST API Server")
    print("="*60)
    
    # Initialize classifier
    classifier = FaceClassifier(PRODUCTION_CONFIG)
    
    if not classifier.load_model():
        print("Failed to load model!")
        return
    
    # Create Flask app
    app = create_app(classifier)
    
    print("\nStarting API server at http://localhost:5000")
    print("Available endpoints:")
    print("  GET  /              - API info")
    print("  GET  /api/health    - Health check")
    print("  POST /api/classify  - Classify image")
    print("  GET  /api/metrics   - Performance metrics")
    print("  GET  /api/config    - Configuration")
    print("\nPress Ctrl+C to stop")
    
    # Run server
    app.run(host='0.0.0.0', port=5000, debug=False)


def run_all_examples():
    """Run all examples interactively."""
    print("\n" + "="*60)
    print("FACE EMOTION CLASSIFICATION EXAMPLES")
    print("="*60)
    print("\nSelect an example to run:")
    print("1. Single Image Classification")
    print("2. Real-time Webcam Classification")
    print("3. Batch Processing")
    print("4. REST API Server")
    print("5. Exit")
    
    while True:
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            example_single_image()
        elif choice == '2':
            example_webcam()
        elif choice == '3':
            example_batch_processing()
        elif choice == '4':
            example_api_server()
        elif choice == '5':
            print("\nExiting...")
            break
        else:
            print("Invalid choice. Please enter 1-5.")


if __name__ == "__main__":
    run_all_examples()
