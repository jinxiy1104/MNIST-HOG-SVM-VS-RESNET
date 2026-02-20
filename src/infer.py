from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from joblib import load
from torchvision.models import resnet18

from .config import HOG_SVM_MODEL_PATH, RESNET_MODEL_PATH
from .features import image28_to_hog

_CACHED = {"loaded": False, "svm": None, "resnet": None, "device": None}


def build_resnet18_1ch() -> nn.Module:
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False,
    )
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def load_models():
    if _CACHED["loaded"]:
        return _CACHED["svm"], _CACHED["resnet"], _CACHED["device"]

    svm_model = load(HOG_SVM_MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(RESNET_MODEL_PATH, map_location=device)
    resnet_model = build_resnet18_1ch()
    resnet_model.load_state_dict(ckpt["state_dict"])
    resnet_model.to(device)
    resnet_model.eval()

    _CACHED["loaded"] = True
    _CACHED["svm"] = svm_model
    _CACHED["resnet"] = resnet_model
    _CACHED["device"] = device
    return svm_model, resnet_model, device


def infer(image28_f32: np.ndarray) -> tuple[int, np.ndarray]:
    if image28_f32.shape != (28, 28):
        raise ValueError(f"Expected shape (28,28), got {image28_f32.shape}")

    svm_model, resnet_model, device = load_models()

    hog_feat = image28_to_hog(image28_f32.astype(np.float32))
    svm_pred = int(svm_model.predict([hog_feat])[0])

    x = torch.from_numpy(image28_f32.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = resnet_model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.float32)
    return svm_pred, probs


if __name__ == "__main__":
    load_models()
    zero = np.zeros((28, 28), dtype=np.float32)
    svm_pred, probs = infer(zero)
    print(f"svm_pred={svm_pred}")
    print(f"resnet_probs_sum={float(probs.sum()):.6f}")
