# 👤 Football Player Face Recognition

A high-performance player identification system using InsightFace and YOLOv8 for professional football match analysis.

## 🎯 Features

- **Multi-Framework Support**: Integration of YOLOv8 for person detection and InsightFace for face identification.
- **State-of-the-Art Models**: Uses RetinaFace for face detection and ArcFace (buffalo_l) for high-accuracy embedding generation.
- **Player Database**: Scalable matching system to identify players from a pre-defined database of profile pictures.
- **Real-time Tracking**: Simple IoU-based tracking to maintain player identities across frames.
- **Robust Detection**: Works in dynamic match conditions by first detecting "Person" boxes and then focusing on face recognition within those regions.

## 📋 Requirements

- Python 3.9+
- OpenCV
- InsightFace
- ONNX Runtime (GPU recommended)
- Ultralytics YOLOv8
- NumPy
- Scipy

## 🚀 Installation

1. Clone this repository:
```bash
git clone https://github.com/Touseeq20/player-face-recognition.git
cd player-face-recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download Models:
   - InsightFace will automatically download the `buffalo_l` model on first run.
   - Place your `yolov8n.pt` model in the root directory.

## 💻 Usage

### 1. Setup Player Database
Create a folder named `tantri` (or as configured) and place player profile pictures named as `PlayerName.jpg`.

### 2. Run Recognition
```bash
python player_face_recognition.py
```

### 3. Configuration
Edit the settings in `player_face_recognition.py`:
```python
KNOWN_FACE_DIR = "tantri"       # Directory with player photos
VIDEO_PATH = "match_video.mp4"  # Input match video
MATCH_THRESHOLD = 25.0          # Distance threshold (lower is stricter)
```

## 🔧 How It Works

1. **Phase 1: Person Detection**:
   - YOLOv8 detects all people in the frame.
   - Filters are applied to focus on players (based on size and aspect ratio).

2. **Phase 2: Face Detection & Embedding**:
   - Inside each person box, RetinaFace detects the face.
   - ArcFace generates a 512-dimensional embedding for the detected face.

3. **Phase 3: Identification**:
   - The embedding is compared against the database using Euclidean distance (or Cosine similarity).
   - If distance < `MATCH_THRESHOLD`, the player name is assigned.

4. **Phase 4: Output**:
   - Overlays player names and distance scores on the video stream.

## 📝 Technical Details

- **Detection**: YOLOv8n (Person) + RetinaFace (Face)
- **Recognition**: ArcFace (InsightFace buffalo_l)
- **Scale**: 512-d embeddings
- **Environment**: Optimized for CUDA/GPU usage via ONNX Runtime.

## 📦 Dependencies

- opencv-python
- insightface
- onnxruntime-gpu
- ultralytics
- numpy
- scipy

## 👤 Author

**Touseeq Ahmed**
- GitHub: [@Touseeq20](https://github.com/Touseeq20)
