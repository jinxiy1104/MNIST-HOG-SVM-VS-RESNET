import numpy as np
import torch
import torch.nn.functional as F


def normalize_image28_u8(image28_u8: np.ndarray) -> np.ndarray:
    if image28_u8.shape != (28, 28):
        raise ValueError(f"Expected image shape (28,28), got {image28_u8.shape}")
    return (image28_u8.astype(np.float32) / 255.0).clip(0.0, 1.0)


def normalize_images28_u8(images28_u8: np.ndarray) -> np.ndarray:
    if images28_u8.ndim != 3 or images28_u8.shape[1:] != (28, 28):
        raise ValueError(f"Expected shape (N,28,28), got {images28_u8.shape}")
    return (images28_u8.astype(np.float32) / 255.0).clip(0.0, 1.0)


def canvas600_to_image28(canvas600: np.ndarray) -> np.ndarray:
    if canvas600.shape != (600, 600):
        raise ValueError(f"Expected canvas shape (600,600), got {canvas600.shape}")

    x = torch.from_numpy(canvas600.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    kernel = torch.tensor(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=torch.float32
    ) / 16.0
    kernel = kernel.view(1, 1, 3, 3)

    x = F.conv2d(x, kernel, padding=1)
    x = F.interpolate(x, size=(28, 28), mode="bilinear", align_corners=False)
    out = x.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
    return np.clip(out, 0.0, 1.0)
