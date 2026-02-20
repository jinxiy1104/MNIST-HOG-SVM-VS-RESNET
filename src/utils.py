import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path | str) -> None:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)


def json_write(path: Path | str, obj: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def json_update(path: Path | str, updates: dict[str, Any]) -> None:
    p = Path(path)
    data: dict[str, Any] = {}
    if p.exists():
        try:
            with p.open("r", encoding="utf-8-sig") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}
    data.update(updates)
    json_write(p, data)
