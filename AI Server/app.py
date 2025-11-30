from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import resnet50

app = Flask(__name__)
CORS(app, resources={
    r"/predict": {
        "origins": "https://sign-language-recognition-production.up.railway.app"
    }
})

def build_resnet50(num_classes: int = 24):
    model = resnet50(weights=None)
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def build_mobilenet_v2(num_classes: int = 24):
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),                     
        nn.Linear(model.last_channel, 512),    
        nn.ReLU(),                            
        nn.Dropout(p=0.5),                    
        nn.Linear(512, num_classes)           
    )
    return model

def get_letter(idx: int) -> str:
    classLabels = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
        5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K',
        10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P',
        15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U',
        20: 'V', 21: 'W', 22: 'X', 23: 'Y'
    }
    return classLabels.get(int(idx), "?")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESNET_PATH = os.path.join(BASE_DIR, "resnet_model.pth")
MOBILENET_PATH = os.path.join(BASE_DIR, "mobilenet_model.pt")

print("BASE_DIR :", BASE_DIR)
print("ResNet   :", RESNET_PATH, "exists?", os.path.exists(RESNET_PATH))
print("MobileNet:", MOBILENET_PATH, "exists?", os.path.exists(MOBILENET_PATH))

device = torch.device("cpu")  

resnet_model = None
mobilenet_model = None

def load_resnet_model(path: str):
    if not os.path.exists(path):
        print(f"[ERROR] Không tìm thấy file ResNet tại: {path}")
        return None

    try:
        print(f"Đang load ResNet (state_dict) từ: {path}")
        state_dict = torch.load(path, map_location=device)
        model = build_resnet50(num_classes=24)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(">> Đã load ResNet thành công!")
        return model
    except Exception as e:
        print("[ERROR] Lỗi khi load ResNet:", e)
        return None


def load_mobilenet_model(path: str):
    if not os.path.exists(path):
        print(f"[WARNING] Không tìm thấy file MobileNet tại: {path}")
        return None
    try:
        print(f"Đang load MobileNet (state_dict) từ: {path}")
        state_dict = torch.load(path, map_location=device)
        model = build_mobilenet_v2(num_classes=24)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(">> Đã load MobileNet thành công!")
        return model
    except Exception as e:
        print("[ERROR] Lỗi khi load MobileNet:", e)
        return None


resnet_model = load_resnet_model(RESNET_PATH)
mobilenet_model = load_mobilenet_model(MOBILENET_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    print("\n" + "=" * 40)
    print("[LOG] Nhận request POST /predict")
    print("ResNet loaded?   ", resnet_model is not None)
    print("MobileNet loaded?", mobilenet_model is not None)

    data = request.get_json()
    if not data or "image" not in data:
        print("[ERROR] Không nhận được ảnh trong JSON")
        return jsonify({"error": "Không nhận được ảnh"}), 400

    try:

        img_base64 = data["image"].split(",")[1]
        img_bytes = base64.b64decode(img_base64)
        img_np = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)  

        if frame is None:
            print("[ERROR] cv2.imdecode trả về None")
            return jsonify({"error": "Không đọc được ảnh"}), 400

        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, (224, 224), interpolation=cv2.INTER_AREA)

        img_norm_gray = img_resized.astype("float32") / 255.0
        img_tensor_gray = torch.from_numpy(img_norm_gray).unsqueeze(0).unsqueeze(0)  
        img_tensor_gray = img_tensor_gray.to(device)

        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)            
        img_norm_rgb = img_rgb.astype(np.float32) / 255.0
        img_tensor_rgb = torch.from_numpy(img_norm_rgb).permute(2, 0, 1)   
        img_tensor_rgb = (img_tensor_rgb - 0.5) / 0.5                      
        img_tensor_rgb = img_tensor_rgb.unsqueeze(0).to(device)           

        resnet_text = "Mô hình ResNet chưa tải"
        if resnet_model is not None:
            with torch.no_grad():
                outputs = resnet_model(img_tensor_gray)   
                probs = F.softmax(outputs, dim=1)
                conf, idx = torch.max(probs, dim=1)
                idx = idx.item()
                conf = conf.item() * 100.0

            letter = get_letter(idx)
            resnet_text = f"{letter} ({conf:.1f}%)"
            print("[ResNet] idx =", idx, "=>", resnet_text)

        mobilenet_text = "Mô hình MobileNet chưa tải"
        if mobilenet_model is not None:
            with torch.no_grad():
                outputs_m = mobilenet_model(img_tensor_rgb)  
                probs_m = F.softmax(outputs_m, dim=1)
                conf_m, idx_m = torch.max(probs_m, dim=1)
                idx_m = idx_m.item()
                conf_m = conf_m.item() * 100.0

            letter_m = get_letter(idx_m)
            mobilenet_text = f"{letter_m} ({conf_m:.1f}%)"
            print("[MobileNet] idx =", idx_m, "=>", mobilenet_text)

        return jsonify({
            "model1": resnet_text,     
            "model2": mobilenet_text   
        })

    except Exception as e:
        print("[ERROR] Exception trong /predict:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Server chạy tại 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)

