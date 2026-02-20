from datetime import datetime, timezone
import os
import threading
import time

import numpy as np
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

from .config import (
    CONFUSION_SVM_PATH,
    HOG_SVM_MODEL_PATH,
    METRICS_PATH,
    MODELS_DIR,
    REPORTS_DIR,
    SEED,
    SVM_C,
    SVM_MAX_ITER,
    TEST_CSV_PATH,
    TRAIN_CSV_PATH,
)
from .eval import compute_accuracy, plot_confusion_matrix
from .features import image28_to_hog
from .io_csv import batch_pixels_to_images28, load_mnist_csv
from .preprocessing import normalize_images28_u8
from .utils import ensure_dir, json_update, set_global_seed


def fit_with_spinner(clf: Pipeline, X: np.ndarray, y: np.ndarray) -> None:
    """Run LinearSVC fit in a worker thread and show indeterminate progress."""
    result: dict[str, BaseException | None] = {"error": None}
    done = threading.Event()

    def _worker() -> None:
        try:
            with threadpool_limits(limits=1):
                clf.fit(X, y)
        except BaseException as exc:  # pragma: no cover
            result["error"] = exc
        finally:
            done.set()

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    t0 = time.perf_counter()
    with tqdm(total=None, desc="Training LinearSVC", unit="tick", leave=True) as pbar:
        while not done.is_set():
            pbar.update(1)
            elapsed = time.perf_counter() - t0
            pbar.set_postfix_str(f"elapsed={elapsed:.1f}s")
            time.sleep(0.2)

    worker.join()
    if result["error"] is not None:
        raise result["error"]


def main() -> None:
    set_global_seed(SEED)
    ensure_dir(MODELS_DIR)
    ensure_dir(REPORTS_DIR)

    X_train_px, y_train = load_mnist_csv(TRAIN_CSV_PATH)
    X_test_px, y_test = load_mnist_csv(TEST_CSV_PATH)

    debug_n = int(os.getenv("DEBUG_TRAIN_SAMPLES", "0"))
    if debug_n > 0:
        X_train_px, y_train = X_train_px[:debug_n], y_train[:debug_n]
        print(f"[INFO] DEBUG_TRAIN_SAMPLES={debug_n}: using subset for training")

    X_train_img = normalize_images28_u8(batch_pixels_to_images28(X_train_px))
    X_test_img = normalize_images28_u8(batch_pixels_to_images28(X_test_px))

    X_train_hog = np.asarray(
        [
        image28_to_hog(img)
        for img in tqdm(X_train_img, total=len(X_train_img), desc="Extracting HOG (train)")
        ],
        dtype=np.float32,
    )
    X_test_hog = np.asarray(
        [
        image28_to_hog(img)
        for img in tqdm(X_test_img, total=len(X_test_img), desc="Extracting HOG (test)")
        ],
        dtype=np.float32,
    )

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svm",
                LinearSVC(
                    C=SVM_C,
                    max_iter=min(SVM_MAX_ITER, 1000),
                    random_state=SEED,
                    dual=False,
                    tol=1e-2,
                    verbose=0,
                ),
            ),
        ]
    )
    print("[INFO] Starting LinearSVC fit...")
    t0 = time.perf_counter()
    fit_with_spinner(clf, X_train_hog, y_train)
    print(f"[INFO] Finished fit in {time.perf_counter() - t0:.2f}s")

    dump(clf, HOG_SVM_MODEL_PATH)

    print("[INFO] Running test prediction...")
    y_pred = clf.predict(X_test_hog)
    acc = compute_accuracy(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred, CONFUSION_SVM_PATH, "HOG + SVM Confusion Matrix")

    json_update(
        METRICS_PATH,
        {
            "svm_test_accuracy": acc,
            "svm_config": {
                "model": "LinearSVC",
                "C": SVM_C,
                "max_iter": SVM_MAX_ITER,
                "hog": {
                    "orientations": 9,
                    "pixels_per_cell": [4, 4],
                    "cells_per_block": [2, 2],
                },
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )

    print(f"SVM test accuracy: {acc:.4f}")
    print(f"Saved model: {HOG_SVM_MODEL_PATH}")
    print(f"Saved confusion matrix: {CONFUSION_SVM_PATH}")
    print(f"Updated metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
