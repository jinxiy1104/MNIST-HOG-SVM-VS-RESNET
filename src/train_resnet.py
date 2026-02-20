from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm.auto import tqdm
from torchvision.models import resnet18

from .config import (
    CONFUSION_RESNET_PATH,
    METRICS_PATH,
    MODELS_DIR,
    REPORTS_DIR,
    RESNET_BATCH_SIZE,
    RESNET_EPOCHS,
    RESNET_LR,
    RESNET_MODEL_PATH,
    RESNET_TRAIN_SIZE,
    RESNET_VAL_SIZE,
    RESNET_WEIGHT_DECAY,
    SEED,
    TEST_CSV_PATH,
    TRAINING_CURVES_PATH,
    TRAIN_CSV_PATH,
)
from .eval import compute_accuracy, plot_confusion_matrix
from .io_csv import batch_pixels_to_images28, load_mnist_csv
from .preprocessing import normalize_images28_u8
from .utils import ensure_dir, json_update, set_global_seed


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


def eval_loader(
    model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="Validation", leave=False):
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == yb).sum().item()
            total += xb.size(0)
    return total_loss / total, total_correct / total


def main() -> None:
    set_global_seed(SEED)
    ensure_dir(MODELS_DIR)
    ensure_dir(REPORTS_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_px, y_train = load_mnist_csv(TRAIN_CSV_PATH)
    X_test_px, y_test = load_mnist_csv(TEST_CSV_PATH)

    X_train = normalize_images28_u8(batch_pixels_to_images28(X_train_px))
    X_test = normalize_images28_u8(batch_pixels_to_images28(X_test_px))

    X_train_t = torch.from_numpy(X_train).unsqueeze(1)
    y_train_t = torch.from_numpy(y_train.astype(np.int64))
    X_test_t = torch.from_numpy(X_test).unsqueeze(1)
    y_test_t = torch.from_numpy(y_test.astype(np.int64))

    full_train = TensorDataset(X_train_t, y_train_t)
    train_ds, val_ds = random_split(
        full_train,
        [RESNET_TRAIN_SIZE, RESNET_VAL_SIZE],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(train_ds, batch_size=RESNET_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=RESNET_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=RESNET_BATCH_SIZE, shuffle=False)

    model = build_resnet18_1ch().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=RESNET_LR, weight_decay=RESNET_WEIGHT_DECAY)

    train_losses: list[float] = []
    val_losses: list[float] = []
    train_accs: list[float] = []
    val_accs: list[float] = []

    for epoch in tqdm(range(1, RESNET_EPOCHS + 1), desc="Epochs"):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for xb, yb in tqdm(
            train_loader,
            desc=f"Train Epoch {epoch}/{RESNET_EPOCHS}",
            leave=False,
        ):
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            running_correct += (preds == yb).sum().item()
            running_total += xb.size(0)

        tr_loss = running_loss / running_total
        tr_acc = running_correct / running_total
        va_loss, va_acc = eval_loader(model, val_loader, device, criterion)

        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        train_accs.append(tr_acc)
        val_accs.append(va_acc)

        print(
            f"Epoch {epoch}/{RESNET_EPOCHS} | "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
        )

    epochs = np.arange(1, RESNET_EPOCHS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_losses, label="train")
    axes[0].plot(epochs, val_losses, label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy")
    axes[0].legend()
    axes[1].plot(epochs, train_accs, label="train")
    axes[1].plot(epochs, val_accs, label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(TRAINING_CURVES_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    model.eval()
    y_pred_list: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in tqdm(test_loader, desc="Testing", leave=False):
            xb = xb.to(device)
            logits = model(xb)
            y_pred_list.append(logits.argmax(dim=1).cpu().numpy())
    y_pred = np.concatenate(y_pred_list)

    test_acc = compute_accuracy(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred, CONFUSION_RESNET_PATH, "ResNet18 Confusion Matrix")

    ckpt = {
        "state_dict": model.state_dict(),
        "model_name": "resnet18_1ch",
        "epoch": RESNET_EPOCHS,
        "config": {
            "epochs": RESNET_EPOCHS,
            "batch_size": RESNET_BATCH_SIZE,
            "lr": RESNET_LR,
            "weight_decay": RESNET_WEIGHT_DECAY,
            "train_size": RESNET_TRAIN_SIZE,
            "val_size": RESNET_VAL_SIZE,
            "seed": SEED,
        },
    }
    torch.save(ckpt, RESNET_MODEL_PATH)

    json_update(
        METRICS_PATH,
        {
            "resnet_test_accuracy": test_acc,
            "resnet_config": ckpt["config"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )

    print(f"ResNet test accuracy: {test_acc:.4f}")
    print(f"Saved checkpoint: {RESNET_MODEL_PATH}")
    print(f"Saved curves: {TRAINING_CURVES_PATH}")
    print(f"Saved confusion matrix: {CONFUSION_RESNET_PATH}")
    print(f"Updated metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
