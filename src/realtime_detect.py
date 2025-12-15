# realtime_detect.py
# Real-time sign language detection using trained model

import cv2
import joblib
import numpy as np
from hand_tracker import HandTracker

# ================= CONFIG =================
MODEL_PATH = "models/sign_model.pkl"
CAMERA_INDEX = 0
# =========================================

# Load trained model and label encoder
data = joblib.load(MODEL_PATH)
model = data["model"]
label_encoder = data["label_encoder"]

tracker = HandTracker()
cap = cv2.VideoCapture(CAMERA_INDEX)

print("Real-time sign detection started")
print("Press Q to quit")

last_prediction = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = tracker.process(rgb)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            landmarks = []

            for lm in hand.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            landmarks_np = np.array(landmarks).reshape(1, -1)

            prediction_index = model.predict(landmarks_np)[0]
            prediction_label = label_encoder.inverse_transform(
                [prediction_index]
            )[0]

            if prediction_label != last_prediction:
                print("Detected:", prediction_label)
                last_prediction = prediction_label

            tracker.draw(frame, hand)

            cv2.putText(
                frame,
                f"Sign: {prediction_label}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

    cv2.imshow("Real-Time Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Detection stopped")
