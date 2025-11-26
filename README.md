# Real-Time Hand Gesture Recognition

A real-time hand gesture recognition system using MediaPipe for hand landmark detection and TensorFlow Lite for lightweight model inference. The system captures gestures from a webcam and classifies them into five categories: **thumb up, thumb down, peace, okay, and fist**.

## Features
- Real-time hand gesture detection via webcam
- 21 landmark-based feature extraction using MediaPipe
- TensorFlow Lite model for fast, on-device inference
- Live annotation of predicted gestures on video feed
- Lightweight and optimized for real-time performance

## Project Structure
gesture_recognition/
├── main.py # Main script for running the system
├── gesture_model.tflite # TensorFlow Lite model file
├── README.md # Project documentation
└── requirements.txt # Python dependencies
## Installation
1. Clone the repository:
```bash
git clone <repo_url>
cd gesture_recognition

Install dependencies:
pip install opencv-python mediapipe tensorflow numpy

