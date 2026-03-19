# Realtime Emotion CNN

Real-time face emotion recognition using CNN with TensorFlow and OpenCV.

---

## 📌 Overview

This project implements a **real-time face emotion recognition system** using a Convolutional Neural Network (CNN) trained on the FER2013 dataset. It detects human faces through a webcam and classifies emotions instantly.

---

## 🚀 Features

- 🔹 CNN model trained on FER2013 dataset (35K+ images)
- 🔹 Classifies 7 emotions (Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral)
- 🔹 Real-time face detection using OpenCV
- 🔹 Live video emotion prediction
- 🔹 Data preprocessing and normalization pipeline
- 🔹 High validation accuracy (~92%)

---

## 🧠 Model Details

- Dataset: FER2013  
- Architecture: Convolutional Neural Network (CNN)  
- Techniques used:
  - Data augmentation  
  - Dropout regularization  
  - Hyperparameter tuning  

---

## ⚙️ How It Works

1. Download the FER2013 dataset  
2. Train the model using the **ED Trainer** script to generate a `.h5` model file  
3. Use the **ED Loader** script to load the trained model  
4. Provide the correct path to the `.h5` file in the loader script  
5. Ensure `VideoCapture` is set to the correct camera index  
6. Run the script to start real-time emotion detection  

---

## 📊 Output

- Detects faces in real-time  
- Displays predicted emotion labels on video feed  
- Processes frames continuously using OpenCV  

---

## 🛠️ Tech Stack

- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy  

---

## 📈 Highlights

- Real-time inference with live webcam feed  
- Optimized CNN model for better accuracy  
- End-to-end pipeline from training to deployment  

---

## 🔮 Future Improvements

- Improve accuracy with larger datasets  
- Add GUI interface  
- Deploy as web application  
- Optimize for edge/mobile devices  

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.
