import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN warnings

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from pathlib import Path

# Configuration
model_dir = Path("C:/Users/AADITYA/Downloads/archive")
model_dir.mkdir(parents=True, exist_ok=True)
MODEL_PATH = model_dir / "gesture_model.tflite"
GESTURE_LABELS = ['thumb_up', 'thumb_down', 'peace', 'okay', 'fist']
INPUT_SHAPE = (1, 21 * 3)  # 21 landmarks * (x,y,z)

def load_model():
    """Load or create the TFLite model"""
    if not MODEL_PATH.exists():
        print("Creating new TFLite model...")
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(63,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        with open(MODEL_PATH, 'wb') as f:
            f.write(tflite_model)
        print(f"Model created at {MODEL_PATH}")
    
    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    return interpreter

def main():
    interpreter = load_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Process landmarks
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    
                    # Run inference
                    input_data = np.array(landmarks, dtype=np.float32).reshape(INPUT_SHAPE)
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    predictions = interpreter.get_tensor(output_details[0]['index'])
                    
                    # Display results
                    predicted_class = np.argmax(predictions)
                    label = f"{GESTURE_LABELS[predicted_class]}"
                    cv2.putText(frame, label, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Hand Gesture Recognition', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()