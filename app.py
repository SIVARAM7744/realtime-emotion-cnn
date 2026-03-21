from __future__ import annotations

import base64
import binascii
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'model.h5'
CASCADE_PATH = BASE_DIR / 'haarcascade_frontalface_default.xml'

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

face_classifier = cv2.CascadeClassifier(str(CASCADE_PATH))
if face_classifier.empty():
    raise RuntimeError(f'Failed to load Haar cascade from {CASCADE_PATH}')

classifier = load_model(str(MODEL_PATH), compile=False)
predict_lock = Lock()

app = FastAPI(title='Emotion CNN', version='1.0.0')
app.mount('/static', StaticFiles(directory=BASE_DIR / 'static'), name='static')


class PredictRequest(BaseModel):
    image: str


def decode_image(data_url: str) -> np.ndarray:
    if ',' in data_url:
        _, encoded = data_url.split(',', 1)
    else:
        encoded = data_url

    try:
        image_bytes = base64.b64decode(encoded)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail='Invalid image payload.') from exc

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail='Could not decode image.')
    return frame


def predict_emotion(frame: np.ndarray) -> dict:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(48, 48),
    )

    if len(faces) == 0:
        return {
            'detected': False,
            'message': 'No face detected. Center your face in the frame and try again.',
        }

    x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
    roi_gray = gray[y:y + h, x:x + w]
    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

    if roi_gray.size == 0 or float(np.sum(roi_gray)) == 0.0:
        return {
            'detected': False,
            'message': 'Face detected, but the image quality is too low for prediction.',
        }

    roi = roi_gray.astype('float32') / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)

    with predict_lock:
        prediction = classifier.predict(roi, verbose=0)[0]

    best_index = int(np.argmax(prediction))
    label = EMOTION_LABELS[best_index]
    confidence = float(prediction[best_index])

    return {
        'detected': True,
        'label': label,
        'confidence': confidence,
        'box': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
        'frame': {'width': int(frame.shape[1]), 'height': int(frame.shape[0])},
        'scores': {
            emotion: float(score)
            for emotion, score in zip(EMOTION_LABELS, prediction)
        },
    }


@app.get('/health')
def health() -> dict:
    return {'status': 'ok'}


@app.get('/')
def index() -> FileResponse:
    return FileResponse(BASE_DIR / 'static' / 'index.html')


@app.post('/predict')
def predict(payload: PredictRequest) -> dict:
    frame = decode_image(payload.image)
    return predict_emotion(frame)


if __name__ == '__main__':
    import os
    import uvicorn

    port = int(os.environ.get('PORT', '8000'))
    uvicorn.run('app:app', host='0.0.0.0', port=port)
