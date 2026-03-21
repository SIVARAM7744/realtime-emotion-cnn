# Emotion CNN

This project is a browser-based facial emotion recognition app built from your trained CNN model. It is now set up for Railway deployment using a FastAPI backend and a React frontend served in the same app.

## Live App Flow

When the Railway link is opened:

- a new browser tab loads the app page
- the browser asks for camera permission
- the frontend opens the user's webcam in the page
- frames are sent to the FastAPI backend
- the backend detects the face, runs the CNN model, and returns the predicted emotion
- the page displays the detected emotion and confidence live

## Project Files

- `emotion-classification-cnn-using-keras.ipynb`: training notebook
- `model.h5`: trained model checkpoint
- `haarcascade_frontalface_default.xml`: Haar cascade face detector
- `main.py`: original local OpenCV webcam script
- `app.py`: FastAPI web app for Railway
- `static/index.html`: frontend entry page
- `static/app.jsx`: React camera UI
- `static/styles.css`: frontend styling
- `requirements.txt`: Python dependencies
- `Procfile`: process start command
- `railway.json`: Railway deployment config

## Model Summary

From the notebook:

- Training images: `28,821`
- Validation images: `7,066`
- Input size: `48x48` grayscale
- Number of emotion classes: `7`
- Best validation accuracy found: `61.75%`
- Highest shown training accuracy before early stopping: `72.34%`

## Stack Used

- FastAPI for backend routes and inference
- React on the frontend for webcam access and live UI
- TensorFlow/Keras for model loading and prediction
- OpenCV for face detection and image preprocessing
- Railway for deployment

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

## Railway Notes

The deployed app uses the browser camera, not a server-side webcam. That is the correct setup for Railway, because Railway containers are headless and cannot open `cv2.imshow()` or access a user's local webcam device directly.

Your original `main.py` is still kept in the project as the local OpenCV version.
