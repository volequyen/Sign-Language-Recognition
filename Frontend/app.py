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
    
    img_base64 = data["image"].split(",")[1]

    img_bytes = base64.b64decode(img_base64)
    img_np = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    resized_img = cv2.resize(frame, (224, 224))

    resized_img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

    resized_img_rgb = resized_img_rgb.astype("float32") / 255.0
    resized_img_rgb = np.expand_dims(resized_img_rgb, axis=0)  # (1,224,224,3)

    # TODO: chạy mô hình ở đây → model.predict(resized_img_rgb)

    model1_result = "Dự đoán Mô hình RESNET: A"
    model2_result = "Dự đoán Mô hình VGG: B"

    return jsonify({
        "model1": model1_result,
        "model2": model2_result
    })


if __name__ == "__main__":
    app.run(debug=True)
