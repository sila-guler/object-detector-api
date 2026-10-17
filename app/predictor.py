import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet50

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class ObjectDetector(nn.Module):
    def __init__(self, baseModel, numClasses):
        super().__init__()
        self.baseModel = baseModel
        self.numClasses = numClasses
        in_f = baseModel.fc.in_features

        self.regressor = nn.Sequential(
            nn.Linear(in_f, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 4), nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_f, 512), nn.ReLU(), nn.Dropout(),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(),
            nn.Linear(512, numClasses),
        )
        self.baseModel.fc = nn.Identity()

    def forward(self, x):
        features = self.baseModel(x)
        return self.regressor(features), self.classifier(features)


with open(MODEL_DIR / "classes.json") as f:
    CLASS_NAMES = json.load(f)

model = ObjectDetector(resnet50(weights=None), len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_DIR / "detector.pth", map_location="cpu"))
model.eval()

normalize = transforms.Normalize(mean=MEAN, std=STD)


def predict(image_bytes: bytes) -> dict:
    """Görüntü baytlarından etiket, olasılık ve kutu koordinatlarını (orijinal piksel) döndürür."""
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Görüntü okunamadı")
    h, w = image.shape[:2]
    rgb = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (224, 224))
    x = normalize(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0).unsqueeze(0)

    with torch.no_grad():
        box, logits = model(x)
        probs = torch.softmax(logits, dim=-1)[0]

    idx = int(probs.argmax())
    sx, sy, ex, ey = box[0].tolist()
    return {
        "label": CLASS_NAMES[idx],
        "probability": round(float(probs[idx]), 4),
        "bbox": [int(sx * w), int(sy * h), int(ex * w), int(ey * h)],
    }
