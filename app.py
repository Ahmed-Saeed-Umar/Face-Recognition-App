from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import cv2
from PIL import Image
import io
import os

app = Flask(__name__)

# ─────────────────────────────────────────────
# Load models at startup
# ─────────────────────────────────────────────
sklearn_model = None
sklearn_scaler = None
keras_model = None

def load_models():
    global sklearn_model, sklearn_scaler, keras_model

    # Load sklearn model
    sk_path = "model/sklearn_model.pkl"
    scaler_path = "model/scaler.pkl"
    if os.path.exists(sk_path):
        with open(sk_path, "rb") as f:
            sklearn_model = pickle.load(f)
        print("✅ Sklearn model loaded.")
    else:
        print("⚠️  Sklearn model not found. Run train_model.py first.")

    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            sklearn_scaler = pickle.load(f)

    # Load Teachable Machine TFLite model
    tflite_path = "model/model.tflite"
    if os.path.exists(tflite_path):
        try:
            import tensorflow as tf
            keras_model = tf.lite.Interpreter(model_path=tflite_path)
            keras_model.allocate_tensors()
            print("✅ Teachable Machine (TFLite) model loaded.")
        except Exception as e:
            print(f"⚠️  Could not load TFLite model: {e}")
    else:
        print("⚠️  TFLite model not found. Place model.tflite in model/ folder.")


def preprocess_for_sklearn(image_bytes, size=(64, 64)):
    """Convert image bytes → flattened grayscale numpy array."""
    img = Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale
    img = img.resize(size)
    arr = np.array(img).flatten().reshape(1, -1)
    return arr


def preprocess_for_keras(image_bytes, size=(224, 224)):
    """Convert image bytes → normalized RGB array for TFLite."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.reshape(1, *arr.shape)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]
    image_bytes = file.read()

    results = {}

    # ── Sklearn prediction ──────────────────────
    if sklearn_model is not None:
        try:
            arr = preprocess_for_sklearn(image_bytes)
            if sklearn_scaler is not None:
                arr = sklearn_scaler.transform(arr)
            pred = sklearn_model.predict(arr)[0]
            prob = sklearn_model.predict_proba(arr)[0]
            confidence = round(float(max(prob)) * 100, 1)
            label = "ME" if pred == 1 else "NOT ME"
            results["sklearn"] = {"label": label, "confidence": confidence}
        except Exception as e:
            results["sklearn"] = {"error": str(e)}
    else:
        results["sklearn"] = {"error": "Model not loaded. Run train_model.py first."}

    # ── TFLite / Teachable Machine prediction ────
    if keras_model is not None:
        try:
            import tensorflow as tf
            input_details  = keras_model.get_input_details()
            output_details = keras_model.get_output_details()

            arr = preprocess_for_keras(image_bytes).astype(np.float32)
            keras_model.set_tensor(input_details[0]['index'], arr)
            keras_model.invoke()

            preds = keras_model.get_tensor(output_details[0]['index'])
            idx = int(np.argmax(preds))
            confidence = round(float(np.max(preds)) * 100, 1)
            label = "ME" if idx == 0 else "NOT ME"
            results["teachable"] = {"label": label, "confidence": confidence}
        except Exception as e:
            results["teachable"] = {"error": str(e)}
    else:
        results["teachable"] = {"error": "TFLite model not loaded."}

    return jsonify(results)


if __name__ == "__main__":
    load_models()
    app.run(debug=True)
