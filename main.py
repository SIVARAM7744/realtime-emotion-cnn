from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.h5"
CASCADE_PATH = BASE_DIR / "haarcascade_frontalface_default.xml"

EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
EMOTION_MESSAGES = {
    "Angry": [
        "Pause, breathe, and let the intensity soften.",
        "You are stronger than this moment.",
    ],
    "Disgust": [
        "Trust what you feel, then respond with clarity.",
        "Protect your peace and move forward.",
    ],
    "Fear": [
        "You are safe in this moment.",
        "Take one steady step at a time with courage.",
    ],
    "Happy": [
        "Your happiness is contagious. Keep shining!",
        "Moments like this are precious, so savor them.",
    ],
    "Neutral": [
        "Calmness is a beautiful strength.",
        "You are balanced, centered, and doing wonderfully.",
    ],
    "Sad": [
        "It is okay to feel sadness. Treat yourself gently today.",
        "Better days are on their way.",
    ],
    "Surprise": [
        "Life's surprises bring unexpected opportunities.",
        "Stay open, stay curious, and embrace the unknown.",
    ],
    "No face detected": [
        "Center your face in the frame so the model can read it clearly.",
    ],
    "Face found": [
        "The face is visible, but the frame quality is too low.",
    ],
}

WINDOW_TITLE = "Real-Time Face Emotion Detection"
WINDOW_BG = (18, 18, 18)
ACCENT = (0, 242, 178)
WHITE = (255, 255, 255)
MUTED = (184, 184, 184)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CANVAS_WIDTH = 980
CANVAS_HEIGHT = 760
FRAME_X = (CANVAS_WIDTH - FRAME_WIDTH) // 2
FRAME_Y = 96


def put_centered_text(
    image: np.ndarray,
    text: str,
    center_x: int,
    y: int,
    font_scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x = max(20, center_x - (text_width // 2))
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def draw_wrapped_lines(
    image: np.ndarray,
    lines: list[str],
    center_x: int,
    start_y: int,
    line_gap: int,
    font_scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    for index, line in enumerate(lines):
        put_centered_text(
            image=image,
            text=line,
            center_x=center_x,
            y=start_y + (index * line_gap),
            font_scale=font_scale,
            color=color,
            thickness=thickness,
        )


def build_ui(frame: np.ndarray, emotion: str, message_lines: list[str]) -> np.ndarray:
    canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), WINDOW_BG, dtype=np.uint8)
    canvas[FRAME_Y : FRAME_Y + FRAME_HEIGHT, FRAME_X : FRAME_X + FRAME_WIDTH] = frame

    put_centered_text(
        image=canvas,
        text="Real-Time Face Emotion Detector",
        center_x=CANVAS_WIDTH // 2,
        y=62,
        font_scale=1.35,
        color=ACCENT,
        thickness=3,
    )

    put_centered_text(
        image=canvas,
        text=f"You are feeling {emotion}",
        center_x=CANVAS_WIDTH // 2,
        y=630,
        font_scale=1.05,
        color=WHITE,
        thickness=2,
    )

    draw_wrapped_lines(
        image=canvas,
        lines=message_lines,
        center_x=CANVAS_WIDTH // 2,
        start_y=685,
        line_gap=34,
        font_scale=0.72,
        color=MUTED,
        thickness=2,
    )

    put_centered_text(
        image=canvas,
        text="Press Q to close",
        center_x=CANVAS_WIDTH // 2,
        y=742,
        font_scale=0.48,
        color=(120, 120, 120),
        thickness=1,
    )

    return canvas


def main() -> None:
    face_classifier = cv2.CascadeClassifier(str(CASCADE_PATH))
    if face_classifier.empty():
        raise RuntimeError(f"Failed to load Haar cascade from {CASCADE_PATH}")

    classifier = load_model(str(MODEL_PATH), compile=False)
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open the webcam. Close other camera apps and try again.")

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, CANVAS_WIDTH, CANVAS_HEIGHT)

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera frame could not be read. Please restart the app.")

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(48, 48),
            )

            emotion = "No face detected"
            message_lines = EMOTION_MESSAGES[emotion]

            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
                cv2.rectangle(frame, (x, y), (x + w, y + h), WHITE, 2)

                roi_gray = gray[y : y + h, x : x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                if roi_gray.size == 0 or float(np.sum(roi_gray)) == 0.0:
                    emotion = "Face found"
                    message_lines = EMOTION_MESSAGES[emotion]
                else:
                    roi = roi_gray.astype("float32") / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    prediction = classifier.predict(roi, verbose=0)[0]
                    emotion = EMOTION_LABELS[int(np.argmax(prediction))]
                    message_lines = EMOTION_MESSAGES[emotion]

            ui = build_ui(frame=frame, emotion=emotion, message_lines=message_lines)
            cv2.imshow(WINDOW_TITLE, ui)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
