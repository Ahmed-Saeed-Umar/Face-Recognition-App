# Face-Recognition-App
Web-based face recognition system using Logistic Regression and Google Teachable Machine (TFLite), integrated with a Flask API.

## Web-Based Visual Recognition System

A Flask web application that classifies facial images as ME or NOT ME using two AI models:

- **Conventional ML Model** — Logistic Regression trained with scikit-learn on grayscale pixel features, achieving 89.7% accuracy.
- **Deep Learning Model** — MobileNetV2-based model trained using Google Teachable Machine and exported as TensorFlow Lite.

Built as part of CSC-350 Artificial Intelligence at Sukkur IBA University.

### Tech Stack
Python, Flask, scikit-learn, TensorFlow Lite, OpenCV, Pillow, React (frontend)

### Features
- Upload any image and get instant ME / NOT ME classification
- Both models run simultaneously with confidence scores
- Color-coded results (green = ME, red = NOT ME)
