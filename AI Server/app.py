from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Cho phép frontend ở port 5000 gọi sang backend 5001
CORS(app, resources={r"/predict": {"origins": "http://127.0.0.1:5000"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "resnet_model.keras")

print("Đang tải mô hình...")
model = None

def load_model_safe(path):
    if not os.path.exists(path):
        print(f"[ERROR] Không tìm thấy file mô hình tại: {path}")
        return None
    try:
        print(f"Đang load model từ: {path}")
        m = load_model(path, compile=False)
        print(">> Đã tải mô hình thành công!")
        return m
    except Exception as e:
        print(f"[ERROR] Lỗi khi tải mô hình: {e}")
        return None

model = load_model_safe(MODEL_PATH)

# Bảng nhãn
labels = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
    6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M',
    12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S',
    18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}

@app.route("/predict", methods=["POST"])
def predict():  
    data = request.get_json()
    if not data or "image" not in data:
        print(" [ERROR] Không nhận được data JSON (data is None)")
        return jsonify({"error": "Không nhận được ảnh"}), 400
    if model is None:
        print(" [ERROR] Model chưa được load!")
        return jsonify({"error": "Mô hình chưa được tải"}), 500

    try:
        img_base64 = data["image"].split(",")[1]
        img_bytes = base64.b64decode(img_base64)
        img_np = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        img = cv2.resize(frame, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)
        idx = int(np.argmax(pred))
        conf = float(np.max(pred) * 100)

        return jsonify({
            "result": f"{labels.get(idx, 'Unknown')} ({conf:.1f}%)"
        })

    except Exception as e:
        print("[ERROR]", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Server chạy tại http://127.0.0.1:5001")
    app.run(debug=False, port=5001)
