# collect_data.py
# Step 1 of the project: Collect hand landmark data for training

import cv2
import csv
import os
from hand_tracker import HandTracker

# ================= CONFIG =================
LABEL = "hello"           # Change this label for each sign you record
SAVE_DIR = "data/raw"     # Folder to store raw CSV files
# =========================================

# Create save directory if it does not exist
os.makedirs(SAVE_DIR, exist_ok=True)

SAVE_PATH = os.path.join(SAVE_DIR, f"{LABEL}.csv")

# Initialize camera and hand tracker
cap = cv2.VideoCapture(0)
tracker = HandTracker()

print("==============================")
print(f"Collecting data for sign: {LABEL}")
print("Show the hand sign clearly")
print("Press Q to stop recording")
print("==============================")

# Open CSV file in append mode
with open(SAVE_PATH, mode="a", newline="") as file:
    writer = csv.writer(file)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = tracker.process(rgb)

        # If a hand is detected
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                landmarks = []

                # Extract 21 hand landmarks (x, y, z)
                for lm in hand.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                # Append label at the end
                landmarks.append(LABEL)

                # Save to CSV
                writer.writerow(landmarks)

                # Draw hand landmarks on screen
                tracker.draw(frame, hand)

        # Instructions on screen
        cv2.putText(
            frame,
            f"Recording sign: {LABEL}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        cv2.putText(
            frame,
            "Press Q to quit",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.imshow("Collect Sign Language Data", frame)

        # Quit when Q is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()

print("==============================")
print("Data collection finished")
print(f"Saved file: {SAVE_PATH}")
print("You can now change LABEL and run again")
print("==============================")