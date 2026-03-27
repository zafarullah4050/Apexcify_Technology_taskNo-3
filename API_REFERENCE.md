# Face Emotion Classification API Reference

Complete API documentation for the face emotion classification module.

## Table of Contents

- [FaceClassifier](#faceclassifier)
- [WebcamClassifier](#webcamclassifier)
- [BatchProcessor](#batchprocessor)
- [Configuration](#configuration)
- [Data Classes](#data-classes)

---

## FaceClassifier

Core classification engine using TensorFlow Lite.

### Constructor

```python
from face_classifier import FaceClassifier, PRODUCTION_CONFIG

classifier = FaceClassifier(config: EmotionClassificationConfig)
```

**Parameters:**
- `config` - Configuration object defining model behavior

**Example:**
```python
from face_classifier import FaceClassifier, PRODUCTION_CONFIG

classifier = FaceClassifier(PRODUCTION_CONFIG)
```

---

### load_model()

Load TFLite model and labels from disk.

```python
success = classifier.load_model()
```

**Returns:** `bool` - True if successful

**Example:**
```python
if not classifier.load_model():
    raise RuntimeError("Failed to load model")
```

---

### classify_image()

Classify a single image (numpy array).

```python
result = classifier.classify_image(image: np.ndarray) -> Optional[ClassificationResult]
```

**Parameters:**
- `image` - Input image in BGR format (OpenCV)

**Returns:** `ClassificationResult` or None

**Example:**
```python
import cv2

image = cv2.imread("photo.jpg")
result = classifier.classify_image(image)

if result:
    print(f"{result.emotion}: {result.confidence:.1%}")
```

---

### classify_image_file()

Classify an image file from disk.

```python
result = classifier.classify_image_file(image_path: str) -> Optional[ClassificationResult]
```

**Parameters:**
- `image_path` - Path to image file

**Returns:** `ClassificationResult` or None

**Example:**
```python
result = classifier.classify_image_file("happy_person.jpg")
print(f"Emotion: {result.emotion}")
```

---

### batch_classify()

Classify multiple images.

```python
results = classifier.batch_classify(
    image_paths: List[str]
) -> List[Tuple[str, Optional[ClassificationResult]]]
```

**Parameters:**
- `image_paths` - List of image file paths

**Returns:** List of (path, result) tuples

**Example:**
```python
image_files = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = classifier.batch_classify(image_files)

for path, result in results:
    if result:
        print(f"{path}: {result.emotion}")
    else:
        print(f"{path}: Failed")
```

---

## WebcamClassifier

Real-time webcam interface with live classification.

### Constructor

```python
from face_classifier import FaceClassifier, WebcamClassifier, REALTIME_CONFIG

classifier = FaceClassifier(REALTIME_CONFIG)
webcam = WebcamClassifier(classifier, webcam_id: int = 0)
```

**Parameters:**
- `classifier` - FaceClassifier instance
- `webcam_id` - Camera device ID (default: 0)

---

### initialize_webcam()

Initialize webcam capture device.

```python
success = webcam.initialize_webcam() -> bool
```

**Returns:** True if successful

**Example:**
```python
if not webcam.initialize_webcam():
    raise RuntimeError("Failed to initialize webcam")
```

---

### run()

Run real-time classification loop.

```python
webcam.run(save_video: bool = False)
```

**Parameters:**
- `save_video` - Whether to save output video file

**Controls:**
- `q` - Quit
- `p` - Pause/Resume
- `s` - Save current frame

**Example:**
```python
webcam.run(save_video=True)  # Saves video file
```

---

### cleanup()

Clean up resources (webcam, windows).

```python
webcam.cleanup()
```

**Called automatically when run() exits.**

---

## BatchProcessor

Process directories of images with statistics.

### Constructor

```python
from face_classifier import FaceClassifier, BatchProcessor, BATCH_CONFIG

classifier = FaceClassifier(BATCH_CONFIG)
processor = BatchProcessor(classifier)
```

---

### process_directory()

Process all images in a directory.

```python
results = processor.process_directory(
    directory: str,
    output_path: Optional[str] = None,
    recursive: bool = True,
) -> Dict
```

**Parameters:**
- `directory` - Directory containing images
- `output_path` - Optional JSON output path
- `recursive` - Search subdirectories

**Returns:** Dictionary with statistics and results

**Example:**
```python
results = processor.process_directory(
    directory="./dataset",
    output_path="results.json",
    recursive=False
)

print(f"Processed {results['statistics']['total']} images")
print(f"Success rate: {results['statistics']['success']/results['statistics']['total']:.1%}")
```

**Output Structure:**
```json
{
  "statistics": {
    "total": 100,
    "success": 95,
    "failed": 5,
    "emotions": {
      "Happy Face": {
        "count": 60,
        "avg_confidence": 0.87
      },
      "Sad Face": {
        "count": 25,
        "avg_confidence": 0.82
      },
      "Neutral": {
        "count": 10,
        "avg_confidence": 0.79
      }
    }
  },
  "results": [
    {
      "path": "image1.jpg",
      "emotion": "Happy Face",
      "confidence": 0.89,
      "all_probabilities": {
        "Happy Face": 0.89,
        "Sad Face": 0.08,
        "Neutral": 0.03
      }
    }
  ]
}
```

---

## Configuration

Predefined configuration profiles.

### PRODUCTION_CONFIG

Best accuracy for production deployment.

```python
from face_classifier import PRODUCTION_CONFIG

classifier = FaceClassifier(PRODUCTION_CONFIG)
```

**Settings:**
- Confidence threshold: 0.75
- Threads: 4
- Show confidence: Yes
- Show FPS: Yes

---

### REALTIME_CONFIG

Low latency for real-time applications.

```python
from face_classifier import REALTIME_CONFIG

classifier = FaceClassifier(REALTIME_CONFIG)
```

**Settings:**
- Confidence threshold: 0.70
- Threads: 2
- TFLite delegates: Disabled (lower overhead)
- Show FPS: Yes

---

### BATCH_CONFIG

Optimized for processing large datasets.

```python
from face_classifier import BATCH_CONFIG

classifier = FaceClassifier(BATCH_CONFIG)
```

**Settings:**
- Confidence threshold: 0.80
- Threads: 8
- Batch size: 32
- Show FPS: No

---

### Custom Configuration

```python
from face_classifier.config import EmotionClassificationConfig

custom_config = EmotionClassificationConfig(
    model_path="./custom_model.tflite",
    labels_path="./custom_labels.txt",
    confidence_threshold=0.85,
    input_size=(299, 299),  # Larger input
    num_threads=6,
    batch_size=16,
    enable_tflite_delegates=True,
    show_confidence=True,
    show_fps=False,
)

classifier = FaceClassifier(custom_config)
```

---

## Data Classes

### ClassificationResult

Result of a classification.

```python
@dataclass
class ClassificationResult:
    emotion: str                    # Predicted emotion label
    confidence: float               # Confidence score (0-1)
    all_probabilities: dict         # All class probabilities
```

**Usage:**
```python
result = classifier.classify_image(image)

print(f"Predicted: {result.emotion}")
print(f"Confidence: {result.confidence:.1%}")
print("All probabilities:")
for emotion, prob in result.all_probabilities.items():
    print(f"  {emotion}: {prob:.1%}")
```

---

### EmotionClassificationConfig

Configuration for the classifier.

```python
@dataclass
class EmotionClassificationConfig:
    model_path: str = "./model.tflite"
    labels_path: str = "./labels.txt"
    confidence_threshold: float = 0.75
    input_size: Tuple[int, int] = (224, 224)
    num_threads: int = 4
    batch_size: int = 1
    enable_tflite_delegates: bool = True
    show_confidence: bool = True
    show_fps: bool = True
    font_scale: float = 1.0
    thickness: int = 2
```

---

## REST API Endpoints

When running the API server (`python main_classifier.py api --port 5000`):

### GET /

API information.

```bash
curl http://localhost:5000
```

---

### GET /api/health

Health check endpoint.

```bash
curl http://localhost:5000/api/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": 1234567890
}
```

---

### POST /api/classify

Classify a single image.

```bash
curl -X POST -F "image=@photo.jpg" http://localhost:5000/api/classify
```

**Request:** Multipart form data with `image` field

**Response:**
```json
{
  "success": true,
  "emotion": "Happy Face",
  "confidence": 0.923,
  "all_probabilities": {
    "Happy Face": 0.923,
    "Sad Face": 0.051,
    "Neutral": 0.026
  },
  "inference_time_ms": 23.5
}
```

---

### GET /api/metrics

Performance metrics.

```bash
curl http://localhost:5000/api/metrics
```

**Response:**
```json
{
  "requests_total": 150,
  "requests_success": 142,
  "requests_failed": 8,
  "success_rate": 0.947,
  "avg_inference_time_ms": 25.3
}
```

---

### GET /api/config

Get current configuration.

```bash
curl http://localhost:5000/api/config
```

**Response:**
```json
{
  "model_path": "./face_emotion_model/model.tflite",
  "labels_path": "./face_emotion_model/labels.txt",
  "confidence_threshold": 0.75,
  "input_size": [224, 224],
  "num_threads": 4
}
```

---

## Error Handling

Always check if operations succeed:

```python
# Model loading
if not classifier.load_model():
    raise RuntimeError("Model failed to load")

# Classification
result = classifier.classify_image_file("photo.jpg")
if result is None:
    print("Classification failed!")
else:
    print(f"Result: {result.emotion}")

# Webcam initialization
if not webcam.initialize_webcam():
    raise RuntimeError("Webcam not available")
```

---

## Complete Examples

### Example 1: Basic Classification

```python
from face_classifier import FaceClassifier, PRODUCTION_CONFIG

# Initialize
classifier = FaceClassifier(PRODUCTION_CONFIG)
classifier.load_model()

# Classify
result = classifier.classify_image_file("person.jpg")

if result:
    print(f"{result.emotion} ({result.confidence:.1%})")
```

---

### Example 2: Real-time Webcam

```python
from face_classifier import FaceClassifier, WebcamClassifier, REALTIME_CONFIG

classifier = FaceClassifier(REALTIME_CONFIG)
classifier.load_model()

webcam = WebcamClassifier(classifier, webcam_id=0)
webcam.initialize_webcam()
webcam.run(save_video=False)
```

---

### Example 3: Batch Processing

```python
from face_classifier import FaceClassifier, BatchProcessor, BATCH_CONFIG
import json

classifier = FaceClassifier(BATCH_CONFIG)
classifier.load_model()

processor = BatchProcessor(classifier)
results = processor.process_directory(
    directory="./dataset",
    output_path="results.json"
)

# Analyze results
stats = results['statistics']
print(f"Total: {stats['total']}")
print(f"Success: {stats['success']}")

for emotion, data in stats['emotions'].items():
    print(f"{emotion}: {data['count']} images, {data['avg_confidence']:.1%} avg confidence")
```

---

### Example 4: Custom Pipeline

```python
from face_classifier import FaceClassifier, PRODUCTION_CONFIG
import cv2

# Load model
classifier = FaceClassifier(PRODUCTION_CONFIG)
classifier.load_model()

# Load and preprocess image
image = cv2.imread("test.jpg")

# Classify
result = classifier.classify_image(image)

# Display result
if result:
    print("="*60)
    print(f"Primary Emotion: {result.emotion}")
    print(f"Confidence: {result.confidence:.1%}")
    print("\nFull Distribution:")
    for emotion, prob in sorted(result.all_probabilities.items(), 
                               key=lambda x: x[1], reverse=True):
        print(f"  {emotion:15s}: {prob:6.1%}")
    print("="*60)
```

---

## Performance Tips

1. **Use GPU acceleration** if available:
   ```python
   config.enable_tflite_delegates = True
   ```

2. **Reduce input size** for faster inference:
   ```python
   config.input_size = (160, 160)  # Instead of (224, 224)
   ```

3. **Adjust thread count** based on CPU:
   ```python
   config.num_threads = 8  # For multi-core CPUs
   ```

4. **Skip frames** in real-time applications:
   ```python
   # Process every 3rd frame
   if frame_count % 3 == 0:
       result = classifier.classify_image(frame)
   ```

---

**Version:** 1.0.0  
**Last Updated:** March 27, 2026
