from flask import Flask, render_template, request, jsonify
import base64
import numpy as np
import cv2

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "image" not in data:
        return jsonify({"error": "No image received"}), 400

    # Lấy phần base64 sau dấu phẩy
    img_base64 = data["image"].split(",")[1]

    # Giải mã base64
    img_bytes = base64.b64decode(img_base64)
    img_np = np.frombuffer(img_bytes, np.uint8)

    # Decode thành ảnh BGR
    frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Cannot decode image"}), 400

    print(">> Ảnh đã gửi lên server (kích thước gốc):", frame.shape)

    # Chuyển sang RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Chuẩn hoá (không resize nữa)
    img_normalized = frame_rgb.astype("float32") / 255.0

    # Nếu model yêu cầu batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)

    print(">> Ảnh sau chuẩn hoá (không resize):", img_batch.shape)
    
    model1_result = "Dự đoán Mô hình RESNET: ..."
    model2_result = "Dự đoán Mô hình MOBILENET: ..."

    return jsonify({
        "model1": model1_result,
        "model2": model2_result
    })

if __name__ == "__main__":
    app.run(debug=True)
