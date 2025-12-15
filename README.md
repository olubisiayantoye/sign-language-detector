# 🤟 Sign Language Detector (Python + OpenCV + MediaPipe)

A beginner‑friendly **real‑time sign language detection system** built with Python.
This project uses **computer vision** and **machine learning** to recognize hand signs from a webcam, display them on screen, and optionally speak them aloud.

Designed to be **simple**, **educational**, and **easy to extend**.

---

## 📌 Features

* 🎥 Real‑time webcam hand tracking
* ✋ Hand landmark detection using **MediaPipe**
* 📊 Data collection and labeling
* 🧠 Machine‑learning sign classification (Scikit‑Learn)
* 🖥 Beautiful desktop GUI (Tkinter)
* 🔊 Optional text‑to‑speech output
* ⚡ Works offline after setup

---

## 🧠 How It Works (Simple Explanation)

Think of the app like a small child learning signs:

1. **Camera sees your hand** 👀
2. **MediaPipe finds your fingers** ✋
3. **Numbers describe finger positions** 🔢
4. **Machine‑learning model learns patterns** 🧠
5. **App predicts the sign name** 🏷
6. **GUI shows (and speaks) the result** 🖥🔊

---

## 🗂 Project Structure

```
sign-language-detector/
│
├── data/
│   └── raw/                 # Collected CSV training data
│
├── models/
│   └── sign_model.pkl       # Trained ML model
│
├── src/
│   ├── collect_data.py      # Collect hand sign data
│   ├── train_model.py       # Train ML model
│   ├── realtime_detect.py   # Detect signs without GUI
│   ├── gui_app.py           # Full GUI application
│   └── hand_tracker.py      # MediaPipe hand tracking logic
│
├── venv/                    # Python virtual environment
├── requirements.txt
└── README.md
```

---

## ⚙ Requirements

* Python **3.10** (recommended)
* Webcam
* Windows / Linux / macOS

### Python Libraries

* opencv‑python
* mediapipe
* numpy
* pandas
* scikit‑learn
* pillow
* pyttsx3
* joblib

---

## 🧪 Installation (Step by Step)

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/sign-language-detector.git
cd sign-language-detector
```

### 2️⃣ Create virtual environment

```powershell
py -3.10 -m venv venv
```

### 3️⃣ Activate virtual environment

```powershell
venv\Scripts\activate
```

### 4️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ✍️ Step 1: Collect Training Data

Edit `collect_data.py`:

```python
LABEL = "hello"
```

Run:

```bash
python src/collect_data.py
```

* Show the same sign repeatedly
* Each frame is saved as training data
* Press **Q** to stop

📁 Output:

```
data/raw/hello.csv
```

Repeat for more signs:

```
hello.csv
thanks.csv
yes.csv
no.csv
```

---

## 🧠 Step 2: Train the Model

Run:

```bash
python src/train_model.py
```

This will:

* Load all CSV files
* Train a classifier
* Save the model

📁 Output:

```
models/sign_model.pkl
```

---

## 🎥 Step 3: Real‑Time Detection (No GUI)

```bash
python src/realtime_detect.py
```

Displays detected sign on webcam feed.

---

## 🖥 Step 4: Run the GUI Application

```bash
python src/gui_app.py
```

### GUI Features

* ▶ Start Camera
* ⏹ Stop Camera
* 🔊 Toggle Voice
* 📌 Live sign detection

---

## 🔊 Text‑to‑Speech

When enabled, detected signs are spoken aloud using `pyttsx3`.

You can turn it ON/OFF from the GUI.

---

## 🧩 Common Problems & Fixes

### ❌ `ModuleNotFoundError: cv2`

Make sure virtual environment is activated:

```powershell
venv\Scripts\activate
pip install opencv-python
```

### ❌ `EOFError: sign_model.pkl`

Model file is empty or corrupted.

✅ Solution:

```bash
python src/train_model.py
```

### ❌ Camera not opening

* Close other apps using the camera
* Change camera index:

```python
cv2.VideoCapture(1)
```

---

## 🚀 Future Improvements

* ✨ Deep learning (TensorFlow / PyTorch)
* 📱 Mobile app version
* 🌍 Multi‑language speech output
* 🧏 Full ASL alphabet support
* 🎨 Improved UI design

---

## 👨‍💻 Author

**Olubisi Ayantoye**
Software Development Student
Brigham Young University–Idaho

---

## 📜 License

This project is open‑source and free to use for learning and research.

---

⭐ If you find this project helpful, please give it a star on GitHub!
