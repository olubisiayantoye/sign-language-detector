import tkinter as tk
from tkinter import ttk
import cv2
import joblib
import pyttsx3
from PIL import Image, ImageTk
from hand_tracker import HandTracker

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Detection System")
        self.root.geometry("900x650")
        self.root.configure(bg="#1e1e2f")

        self.tracker = HandTracker()
        self.model = joblib.load("models/sign_model.pkl")

        # Dummy model for testing before training
        class DummyModel:
           def predict(self, X):
              return ["Hello"]  # always returns 'Hello' for testing

        self.model = DummyModel()



        self.engine = pyttsx3.init()
        self.voice_enabled = True
        self.last_word = ""
        self.cap = None

        self.build_ui()

    def build_ui(self):
        title = tk.Label(
            self.root,
            text="Real-Time Sign Language Detection",
            font=("Segoe UI", 22, "bold"),
            fg="white",
            bg="#1e1e2f"
        )
        title.pack(pady=15)

        self.video_label = tk.Label(self.root, bg="#000000")
        self.video_label.pack(pady=10)

        self.result_label = tk.Label(
            self.root,
            text="Detected Sign: ---",
            font=("Segoe UI", 18),
            fg="#00ffcc",
            bg="#1e1e2f"
        )
        self.result_label.pack(pady=10)

        btn_frame = tk.Frame(self.root, bg="#1e1e2f")
        btn_frame.pack(pady=20)

        ttk.Button(btn_frame, text="Start Camera", command=self.start_camera).grid(row=0, column=0, padx=10)
        ttk.Button(btn_frame, text="Stop Camera", command=self.stop_camera).grid(row=0, column=1, padx=10)
        ttk.Button(btn_frame, text="Toggle Voice", command=self.toggle_voice).grid(row=0, column=2, padx=10)

        self.status = tk.Label(
            self.root,
            text="Status: Idle",
            font=("Segoe UI", 12),
            fg="white",
            bg="#1e1e2f"
        )
        self.status.pack(pady=10)

    def start_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.status.config(text="Status: Camera Running")
            self.update_frame()

    def stop_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None
            self.video_label.config(image="")
            self.status.config(text="Status: Camera Stopped")

    def toggle_voice(self):
        self.voice_enabled = not self.voice_enabled
        self.status.config(text=f"Voice: {'ON' if self.voice_enabled else 'OFF'}")

    def update_frame(self):
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.tracker.process(rgb)

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                prediction = self.model.predict([landmarks])[0]
                self.result_label.config(text=f"Detected Sign: {prediction}")

                if prediction != self.last_word and self.voice_enabled:
                    self.engine.say(prediction)
                    self.engine.runAndWait()
                    self.last_word = prediction

                self.tracker.draw(frame, hand)

        img = Image.fromarray(rgb)
        img = img.resize((640, 480))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

        self.root.after(10, self.update_frame)


if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()
