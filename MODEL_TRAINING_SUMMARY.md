# Model Training Summary

## Project Goal

Build a convolutional neural network that detects human facial emotion from face images and then use the trained model in a real-time webcam emotion detector.

## Training Setup Found In The Notebook

Notebook file:

- `emotion-classification-cnn-using-keras.ipynb`

Dataset setup used in the notebook:

- Folder path: `../input/face-expression-recognition-dataset/images/`
- Train directory: `train`
- Validation directory: `validation`
- Total training images: `28,821`
- Total validation images: `7,066`
- Emotion classes: `7`
- Input image size: `48x48`
- Image mode: grayscale
- Batch size: `128`

## CNN Architecture Used

The model is a Sequential CNN with:

1. Conv2D `64`, kernel `3x3`, same padding
2. BatchNormalization
3. ReLU
4. MaxPooling2D
5. Dropout `0.25`
6. Conv2D `128`, kernel `5x5`, same padding
7. BatchNormalization
8. ReLU
9. MaxPooling2D
10. Dropout `0.25`
11. Conv2D `512`, kernel `3x3`, same padding
12. BatchNormalization
13. ReLU
14. MaxPooling2D
15. Dropout `0.25`
16. Conv2D `512`, kernel `3x3`, same padding
17. BatchNormalization
18. ReLU
19. MaxPooling2D
20. Dropout `0.25`
21. Flatten
22. Dense `256`
23. BatchNormalization
24. ReLU
25. Dropout `0.25`
26. Dense `512`
27. BatchNormalization
28. ReLU
29. Dropout `0.25`
30. Dense `7`, softmax

Approximate parameter count:

- `4,478,727` total parameters

## Training Configuration

- Optimizer in final training cell: `Adam(lr=0.001)`
- Loss: `categorical_crossentropy`
- Metric: `accuracy`
- Epochs set: `48`
- Checkpoint: save best model to `./model.h5`
- EarlyStopping: monitor `val_loss`, patience `3`, restore best weights
- ReduceLROnPlateau: enabled

## Accuracy Found

From the recorded notebook output:

- Epoch 1 validation accuracy: `34.82%`
- Epoch 4 validation accuracy: `54.91%`
- Epoch 9 validation accuracy: `60.38%`
- Epoch 11 validation accuracy: `60.81%`
- Epoch 13 validation accuracy: `61.75%`

Best validation accuracy reached:

- `61.75%`

Highest training accuracy shown before early stopping:

- `72.34%`

Training stopped at epoch 14 and restored the best checkpoint.

## Inference Pipeline

After training:

- the saved model is loaded from `model.h5`
- OpenCV detects faces using `haarcascade_frontalface_default.xml`
- each detected face is converted to a grayscale `48x48` ROI
- pixel values are normalized to `[0, 1]`
- the model predicts one of 7 emotions
- the predicted label is drawn above the face in the webcam frame

## Files That Show The Final Work

- `emotion-classification-cnn-using-keras.ipynb`: training workflow
- `model.h5`: trained model
- `main.py`: real-time prediction script
- `outputs/`: sample result images

## Important Deployment Note

The current app is designed for local webcam use. It is not directly deployable to Railway as a visible webcam app because Railway does not provide a local camera device or desktop windowing for `cv2.imshow()`.

For Railway, the cleanest next version would be a small upload-based web app or prediction API built around the existing `model.h5`.
