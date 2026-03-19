from pathlib import Path
import math

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from tabulate import tabulate


PROJECT_ROOT = Path(__file__).parent.parent
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
IMAGES_DIR = CLEANED_DIR / "images"
ANNOTATION_FILE = CLEANED_DIR / "affectnet_annotations.csv"


def load_split(split_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load flattened grayscale images and labels for a given split."""
    df = pd.read_csv(ANNOTATION_FILE)
    df = df[df["split"] == split_name].copy()
    img_dir = IMAGES_DIR / split_name

    X = np.zeros((len(df), 96 * 96), dtype=np.float32)
    y = df["label"].astype(int).to_numpy()
    filenames = df["file_name"].tolist()

    for i, filename in enumerate(tqdm(filenames, desc=f"Loading {split_name} images")):
        img = Image.open(img_dir / filename).convert("L")
        X[i] = np.asarray(img, dtype=np.float32).reshape(-1)
    return X, y


def subsample(X: np.ndarray, y: np.ndarray, max_per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Balanced per-class subsampling for faster experimentation."""
    rng = np.random.default_rng(seed)
    keep_idx = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, max_per_class, replace=False)
        keep_idx.extend(idx.tolist())
    keep_idx = np.array(sorted(keep_idx))
    return X[keep_idx], y[keep_idx]


def build_gabor_kernel(
    size: int,
    sigma: float,
    theta: float,
    lambd: float,
    gamma: float = 0.5,
    psi: float = 0.0,
) -> np.ndarray:
    """Construct a real-valued Gabor kernel from the analytic formula."""
    half = size // 2
    y, x = np.mgrid[-half: half + 1, -half: half + 1]

    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gauss = np.exp(-((x_theta ** 2 + (gamma ** 2) * (y_theta ** 2)) / (2.0 * sigma ** 2)))
    wave = np.cos((2.0 * np.pi * x_theta / lambd) + psi)
    kernel = gauss * wave

    kernel = kernel - np.mean(kernel)
    norm = np.linalg.norm(kernel)
    if norm > 0:
        kernel = kernel / norm
    return kernel.astype(np.float32)


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2D convolution implemented with NumPy sliding windows."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode="reflect")

    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw))
    return np.einsum("ijkl,kl->ij", windows, kernel, optimize=True).astype(np.float32)


def build_gabor_bank() -> list[np.ndarray]:
    """Create a compact Gabor filter bank for energy features."""
    kernels = []
    orientations = [0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0]
    wavelengths = [4.0, 8.0]
    sigmas = [2.0, 4.0]
    size = 15

    for theta in orientations:
        for lambd in wavelengths:
            for sigma in sigmas:
                kernels.append(build_gabor_kernel(size=size, sigma=sigma, theta=theta, lambd=lambd))
    return kernels


def extract_gabor_energy_features(X_flat: np.ndarray, kernels: list[np.ndarray]) -> np.ndarray:
    """Extract one energy value per Gabor kernel for each image."""
    n_samples = X_flat.shape[0]
    n_features = len(kernels)
    feats = np.zeros((n_samples, n_features), dtype=np.float32)

    for i in tqdm(range(n_samples), desc="Extracting Gabor energy features"):
        img = X_flat[i].reshape(96, 96).astype(np.float32)
        for j, kernel in enumerate(kernels):
            response = convolve2d(img, kernel)
            feats[i, j] = float(np.mean(response ** 2))
    return feats


class KNearestNeighbors:
    """KNN classifier from scratch with selectable distance metric."""

    def __init__(self, k: int = 5, distance_metric: str = "euclidean"):
        if k < 1:
            raise ValueError("k must be >= 1")
        valid_metrics = {"euclidean", "manhattan", "cosine"}
        if distance_metric not in valid_metrics:
            raise ValueError(f"distance_metric must be one of {sorted(valid_metrics)}")

        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
        self.classes_ = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.X_train = np.asarray(X_train, dtype=np.float32)
        self.y_train = np.asarray(y_train, dtype=int)
        self.classes_ = np.unique(self.y_train)

    def _pairwise_distance(self, X_test: np.ndarray) -> np.ndarray:
        X_test = np.asarray(X_test, dtype=np.float32)
        X_train = self.X_train

        if self.distance_metric == "euclidean":
            test_sq = np.sum(X_test ** 2, axis=1, keepdims=True)
            train_sq = np.sum(X_train ** 2, axis=1, keepdims=True).T
            dist_sq = np.maximum(test_sq + train_sq - 2.0 * (X_test @ X_train.T), 0.0)
            return np.sqrt(dist_sq, dtype=np.float32)

        if self.distance_metric == "manhattan":
            return np.sum(np.abs(X_test[:, None, :] - X_train[None, :, :]), axis=2, dtype=np.float32)

        # cosine distance = 1 - cosine similarity
        eps = 1e-8
        test_norm = X_test / (np.linalg.norm(X_test, axis=1, keepdims=True) + eps)
        train_norm = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + eps)
        sim = test_norm @ train_norm.T
        return (1.0 - sim).astype(np.float32)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        distances = self._pairwise_distance(X_test)
        k = min(self.k, self.X_train.shape[0])
        nn_idx = np.argsort(distances, axis=1)[:, :k]
        nn_labels = self.y_train[nn_idx]

        preds = np.zeros(nn_labels.shape[0], dtype=int)
        for i in range(nn_labels.shape[0]):
            labels, counts = np.unique(nn_labels[i], return_counts=True)
            preds[i] = int(labels[np.argmax(counts)])
        return preds

    def predict_proba(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        distances = self._pairwise_distance(X_test)
        k = min(self.k, self.X_train.shape[0])
        nn_idx = np.argsort(distances, axis=1)[:, :k]
        nn_labels = self.y_train[nn_idx]

        probs = np.zeros((X_test.shape[0], len(self.classes_)), dtype=np.float32)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        for i in range(X_test.shape[0]):
            for c in nn_labels[i]:
                probs[i, class_to_idx[int(c)]] += 1.0
        probs /= float(k)
        return probs, self.classes_


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> float:
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=np.int32)
    for yt, yp in zip(y_true, y_pred):
        cm[label_to_idx[int(yt)], label_to_idx[int(yp)]] += 1

    f1s = []
    for i in range(len(labels)):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def tune_knn_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    k_values: list[int],
    distance_metrics: list[str],
) -> tuple[KNearestNeighbors, dict[str, float], list[list[str]]]:
    """Grid search over KNN hyperparameters using validation accuracy."""
    best_model = None
    best_config = None
    best_acc = -math.inf
    rows = []
    labels = np.unique(np.concatenate([y_train, y_val]))

    for metric in distance_metrics:
        for k in k_values:
            model = KNearestNeighbors(k=k, distance_metric=metric)
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            val_f1 = macro_f1_score(y_val, val_pred, labels)

            rows.append([str(k), metric, f"{val_acc:.4f}", f"{val_f1:.4f}"])
            if val_acc > best_acc:
                best_acc = val_acc
                best_model = model
                best_config = {"k": k, "distance_metric": metric, "val_accuracy": val_acc, "val_macro_f1": val_f1}

    return best_model, best_config, rows


def main() -> None:
    print("=== Gabor Energy + KNN (from scratch) ===")
    print("Loading dataset splits...")
    X_train_raw, y_train_raw = load_split("train")
    X_val_raw, y_val_raw = load_split("val")
    X_test_raw, y_test_raw = load_split("test")

    # Balanced subset keeps runtime practical while preserving class coverage.
    X_train_raw, y_train_raw = subsample(X_train_raw, y_train_raw, max_per_class=200, seed=42)
    X_val_raw, y_val_raw = subsample(X_val_raw, y_val_raw, max_per_class=100, seed=43)
    X_test_raw, y_test_raw = subsample(X_test_raw, y_test_raw, max_per_class=100, seed=44)

    print("Building Gabor bank...")
    gabor_kernels = build_gabor_bank()
    print(f"Total Gabor filters: {len(gabor_kernels)}")

    Z_train = extract_gabor_energy_features(X_train_raw, gabor_kernels)
    Z_val = extract_gabor_energy_features(X_val_raw, gabor_kernels)
    Z_test = extract_gabor_energy_features(X_test_raw, gabor_kernels)

    mean = Z_train.mean(axis=0)
    std = Z_train.std(axis=0)
    std[std == 0] = 1.0
    Z_train = (Z_train - mean) / std
    Z_val = (Z_val - mean) / std
    Z_test = (Z_test - mean) / std

    k_values = [1, 3, 5, 7, 9, 11, 15]
    distance_metrics = ["euclidean", "manhattan", "cosine"]
    best_model, best_config, search_rows = tune_knn_hyperparameters(
        Z_train, y_train_raw, Z_val, y_val_raw, k_values, distance_metrics
    )

    print("\nHyperparameter search results:")
    print(tabulate(search_rows, headers=["k", "distance_metric", "val_accuracy", "val_macro_f1"], tablefmt="grid"))
    print("\nBest config:", best_config)

    # Retrain best config on train+val and evaluate on test.
    X_trainval = np.vstack([Z_train, Z_val])
    y_trainval = np.concatenate([y_train_raw, y_val_raw])
    final_model = KNearestNeighbors(k=best_config["k"], distance_metric=best_config["distance_metric"])
    final_model.fit(X_trainval, y_trainval)
    test_pred = final_model.predict(Z_test)

    labels = np.unique(np.concatenate([y_trainval, y_test_raw]))
    test_acc = accuracy_score(y_test_raw, test_pred)
    test_macro_f1 = macro_f1_score(y_test_raw, test_pred, labels)

    print("\nFinal test metrics:")
    print(tabulate(
        [["Gabor+KNN", f"{test_acc:.4f}", f"{test_macro_f1:.4f}", str(best_config["k"]), best_config["distance_metric"]]],
        headers=["model", "test_accuracy", "test_macro_f1", "best_k", "best_metric"],
        tablefmt="grid",
    ))


if __name__ == "__main__":
    main()
