"""
Test setup and validate face classifier module.

Checks:
- Dependencies installed
- Model files present
- Labels file valid
- Basic functionality works
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n" + "="*60)
    print("CHECKING DEPENDENCIES")
    print("="*60)
    
    success = True
    packages = {
        'opencv-python': 'cv2',
        'tensorflow': 'tensorflow',
        'numpy': 'numpy',
        'pillow': 'PIL',
        'flask': 'flask',
        'tqdm': 'tqdm',
    }
    
    for package_name, import_name in packages.items():
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {package_name}: {version}")
        except ImportError:
            print(f"✗ {package_name}: NOT INSTALLED")
            success = False
    
    return success


def check_model_files():
    """Check if model files exist."""
    print("\n" + "="*60)
    print("CHECKING MODEL FILES")
    print("="*60)
    
    model_path = Path("./face_emotion_model/model.tflite")
    labels_path = Path("./face_emotion_model/labels.txt")
    
    success = True
    
    # Check model
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✓ Model file: {model_path} ({size_mb:.1f} MB)")
    else:
        print(f"✗ Model file NOT found: {model_path}")
        print("\n  To fix this:")
        print("  1. Visit https://teachablemachine.withgoogle.com")
        print("  2. Create an Image Project with 3 classes:")
        print("     - Happy Face")
        print("     - Sad Face")
        print("     - Neutral")
        print("  3. Train and export as TensorFlow Lite")
        print("  4. Download and extract to ./face_emotion_model/")
        success = False
    
    # Check labels
    if labels_path.exists():
        print(f"✓ Labels file: {labels_path}")
        with open(labels_path, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        print(f"  Classes ({len(labels)}): {labels}")
        
        # Validate expected labels
        expected = ["Happy Face", "Sad Face", "Neutral"]
        if all(any(exp in lab for lab in labels) for exp in expected):
            print(f"  ✓ All expected emotion labels found")
        else:
            print(f"  ⚠ Labels don't match expected emotions")
            print(f"     Expected: {expected}")
    else:
        print(f"✗ Labels file NOT found: {labels_path}")
        success = False
    
    return success


def check_test_images():
    """Check if test images are available."""
    print("\n" + "="*60)
    print("CHECKING TEST IMAGES")
    print("="*60)
    
    images_dir = Path("./images")
    
    if not images_dir.exists():
        print(f"⚠ Images directory not found: {images_dir}")
        print("  Creating directory...")
        images_dir.mkdir(parents=True, exist_ok=True)
        return False
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']:
        image_files.extend(images_dir.glob(ext))
    
    if image_files:
        print(f"✓ Found {len(image_files)} test images:")
        for img in image_files[:5]:  # Show first 5
            size_kb = img.stat().st_size / 1024
            print(f"  - {img.name} ({size_kb:.1f} KB)")
        
        if len(image_files) > 5:
            print(f"  ... and {len(image_files) - 5} more")
        
        return True
    else:
        print(f"⚠ No test images found in {images_dir}")
        print("  Add some test images to this directory")
        return False


def test_classifier_import():
    """Test importing the classifier module."""
    print("\n" + "="*60)
    print("TESTING MODULE IMPORT")
    print("="*60)
    
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from face_classifier import FaceClassifier, PRODUCTION_CONFIG
        print("✓ Successfully imported face_classifier module")
        print(f"  Configuration: PRODUCTION_CONFIG")
        print(f"  Input size: {PRODUCTION_CONFIG.input_size}")
        print(f"  Confidence threshold: {PRODUCTION_CONFIG.confidence_threshold}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import module: {e}")
        return False


def test_model_loading():
    """Test loading the TFLite model."""
    print("\n" + "="*60)
    print("TESTING MODEL LOADING")
    print("="*60)
    
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from face_classifier import FaceClassifier, PRODUCTION_CONFIG
        
        classifier = FaceClassifier(PRODUCTION_CONFIG)
        
        if classifier.load_model():
            print("✓ Model loaded successfully")
            print(f"  Input shape: {classifier.input_details[0]['shape']}")
            print(f"  Output shape: {classifier.output_details[0]['shape']}")
            print(f"  Number of classes: {len(classifier.labels)}")
            print(f"  Classes: {classifier.labels}")
            return True
        else:
            print("✗ Failed to load model")
            print("  See error logs above for details")
            return False
            
    except Exception as e:
        print(f"✗ Error during model loading: {e}")
        logger.debug("Detailed error:", exc_info=True)
        return False


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "="*70)
    print("FACE EMOTION CLASSIFIER - SETUP VALIDATION")
    print("="*70)
    
    results = {
        "Dependencies": check_dependencies(),
        "Model Files": check_model_files(),
        "Test Images": check_test_images(),
        "Module Import": test_classifier_import(),
        "Model Loading": test_model_loading(),
    }
    
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED! Setup is complete.")
        print("\nYou can now:")
        print("  1. Run examples: python main_classifier.py examples")
        print("  2. Classify images: python main_classifier.py classify --image photo.jpg")
        print("  3. Start webcam: python main_classifier.py webcam")
        print("  4. Start API server: python main_classifier.py api --port 5000")
    else:
        print("\n⚠ Some tests failed. Please review the output above.")
        print("\nCommon fixes:")
        print("  - Install dependencies: pip install -r face_classifier/requirements.txt")
        print("  - Download Teachable Machine model to ./face_emotion_model/")
        print("  - Add test images to ./images/ directory")
    
    print("="*70 + "\n")
    
    return all(results.values())


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
