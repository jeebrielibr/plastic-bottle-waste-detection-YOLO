import base64
import os

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "v8n_colab", "best.pt")
CONF_THRESHOLD = 0.5

model = YOLO(MODEL_PATH)


def run_detection(img_array):
    results = model.predict(source=img_array, conf=CONF_THRESHOLD, verbose=False)
    annotated = results[0].plot()

    detections = []
    for box in results[0].boxes:
        detections.append(
            {
                "confidence": round(float(box.conf[0]), 4),
                "bbox": [round(float(v), 2) for v in box.xyxy[0].tolist()],
            }
        )

    return annotated, detections


def encode_image(img_array):
    _, buffer = cv2.imencode(".jpg", img_array)
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")


def decode_base64_image(data_url):
    _, encoded = data_url.split(",", 1)
    decoded = base64.b64decode(encoded)
    nparr = np.frombuffer(decoded, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect/image", methods=["POST"])
def detect_image():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No image selected"}), 400

    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"success": False, "error": "Invalid image"}), 400

    annotated, detections = run_detection(img)

    return jsonify(
        {
            "success": True,
            "image": encode_image(annotated),
            "count": len(detections),
            "detections": detections,
        }
    )


@app.route("/detect/frame", methods=["POST"])
def detect_frame():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"success": False, "error": "No frame provided"}), 400

    img = decode_base64_image(data["image"])

    if img is None:
        return jsonify({"success": False, "error": "Invalid frame"}), 400

    annotated, detections = run_detection(img)

    return jsonify(
        {
            "success": True,
            "image": encode_image(annotated),
            "count": len(detections),
            "detections": detections,
        }
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
