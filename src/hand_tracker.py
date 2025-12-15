import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=2,
                                         min_detection_confidence=0.7,
                                         min_tracking_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils

    def process(self, frame):
        """Process the frame and return results"""
        return self.hands.process(frame)

    def draw(self, frame, hand_landmarks):
        """Draw hand landmarks on the frame"""
        self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
