from __future__ import annotations

from pathlib import Path
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.h5"
CASCADE_PATH = BASE_DIR / "haarcascade_frontalface_default.xml"

EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
EMOTION_MESSAGES = {
    "Angry": "Pause, breathe, and let the intensity soften.\nYou are stronger than this moment.",
    "Disgust": "Trust what you feel, then respond with clarity.\nProtect your peace and move forward.",
    "Fear": "You are safe in this moment.\nTake one steady step at a time with courage.",
    "Happy": "Your happiness is contagious. Keep shining!\nMoments like this are precious, so savor them.",
    "Neutral": "Calmness is a beautiful strength.\nYou are balanced, centered, and doing wonderfully.",
    "Sad": "It is okay to feel sadness. Treat yourself gently today.\nBetter days are on their way.",
    "Surprise": "Life's surprises bring unexpected opportunities.\nStay open, stay curious, and embrace the unknown.",
}

WINDOW_BG = "#121212"
ACCENT = "#00f2b2"
TEXT_PRIMARY = "#ffffff"
TEXT_SECONDARY = "#b8b8b8"
BOX_COLOR = (255, 255, 255)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


class EmotionApp:
    def __init__(self) -> None:
        self.face_classifier = cv2.CascadeClassifier(str(CASCADE_PATH))
        if self.face_classifier.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {CASCADE_PATH}")

        self.classifier = load_model(str(MODEL_PATH), compile=False)
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            raise RuntimeError("Could not open the webcam. Close other camera apps and try again.")

        self.root = tk.Tk()
        self.root.title("Real-Time Face Emotion Detection")
        self.root.configure(bg=WINDOW_BG)
        self.root.geometry("980x760")
        self.root.minsize(900, 700)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.video_photo: ImageTk.PhotoImage | None = None
        self.current_emotion = "Waiting for face..."

        self._build_ui()
        self._update_frame()

    def _build_ui(self) -> None:
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        header = tk.Label(
            self.root,
            text="Real-Time Face Emotion Detector",
            font=("Segoe UI", 30, "bold"),
            fg=ACCENT,
            bg=WINDOW_BG,
        )
        header.grid(row=0, column=0, pady=(24, 18))

        self.video_label = tk.Label(
            self.root,
            bg=WINDOW_BG,
            bd=0,
            highlightthickness=0,
        )
        self.video_label.grid(row=1, column=0, padx=24, sticky="n")

        self.emotion_label = tk.Label(
            self.root,
            text="You are feeling Waiting...",
            font=("Segoe UI", 26, "bold"),
            fg=TEXT_PRIMARY,
            bg=WINDOW_BG,
        )
        self.emotion_label.grid(row=2, column=0, pady=(28, 10))

        self.message_label = tk.Label(
            self.root,
            text="Center your face in the camera to begin detection.",
            font=("Segoe UI", 18),
            fg=TEXT_SECONDARY,
            bg=WINDOW_BG,
            justify="center",
            wraplength=760,
        )
        self.message_label.grid(row=3, column=0, padx=32, pady=(0, 22))

        self.footer_label = tk.Label(
            self.root,
            text="Press Q inside the window or close it to exit.",
            font=("Segoe UI", 11),
            fg="#7d7d7d",
            bg=WINDOW_BG,
        )
        self.footer_label.grid(row=4, column=0, pady=(0, 18))

        self.root.bind("<KeyPress-q>", lambda _event: self.close())
        self.root.bind("<KeyPress-Q>", lambda _event: self.close())

    def _predict_emotion(self, frame: np.ndarray) -> tuple[np.ndarray, str, str]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(48, 48),
        )

        display_frame = frame.copy()
        emotion = "No face detected"
        message = "Center your face in the frame so the model can read it clearly."

        if len(faces) == 0:
            return display_frame, emotion, message

        x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), BOX_COLOR, 2)

        roi_gray = gray[y : y + h, x : x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if roi_gray.size == 0 or float(np.sum(roi_gray)) == 0.0:
            return display_frame, "Face found", "The face is visible, but the frame quality is too low."

        roi = roi_gray.astype("float32") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        prediction = self.classifier.predict(roi, verbose=0)[0]
        label = EMOTION_LABELS[int(np.argmax(prediction))]
        emotion = label
        message = EMOTION_MESSAGES[label]

        return display_frame, emotion, message

    def _update_frame(self) -> None:
        if not self.camera or not self.camera.isOpened():
            return

        ok, frame = self.camera.read()
        if not ok:
            self.message_label.config(text="Camera frame could not be read. Please restart the app.")
            self.root.after(120, self._update_frame)
            return

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        processed_frame, emotion, message = self._predict_emotion(frame)

        image_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_rgb)
        self.video_photo = ImageTk.PhotoImage(image=image)
        self.video_label.config(image=self.video_photo)

        self.current_emotion = emotion
        self.emotion_label.config(text=f"You are feeling {emotion}")
        self.message_label.config(text=message)

        self.root.after(90, self._update_frame)

    def close(self) -> None:
        if self.camera and self.camera.isOpened():
            self.camera.release()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    app = EmotionApp()
    app.run()
