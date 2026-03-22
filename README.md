# Real-Time Face Emotion Detector

This repository is set up as a local desktop emotion detection project using your trained `model.h5` file.

The app opens your webcam, detects the largest face in the frame, draws the face box, predicts the emotion with the trained CNN, and shows the result in a dark desktop interface like the reference output images in [`outputs/`](outputs/).

## Files

- `main.py`: local desktop app for webcam-based emotion detection
- `model.h5`: trained CNN model used for prediction
- `haarcascade_frontalface_default.xml`: Haar cascade for face detection
- `outputs/`: sample screenshots of the desired interface/output
- `requirements.txt`: Python dependencies for local setup
- `README.md`: local setup and run instructions

## Run Locally

Install dependencies:

```bash
py -3.11 -m pip install -r requirements.txt
```

Start the desktop app:

```bash
py -3.11 main.py
```

Close the app with the window close button or press `Q`.

## Notes

- `model.h5` is loaded directly from this repository, so keep it in the project root.
- This project is intended for local use and GitHub upload only. No deployment setup is required.
- Python 3.11 is the recommended version for this project on Windows.
