from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from PCA import PCA

PROJECT_ROOT = Path(__file__).parent.parent
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
IMAGES_DIR = CLEANED_DIR / "images"
ANNOTATION_FILE = CLEANED_DIR / "affectnet_annotations.csv"
PCA_CACHE_FILE = PROJECT_ROOT / "script" / "cache" / "pca.pkl"

class KernelFunctions:
    @staticmethod
    def linear(X1, X2):
        return (X1 @ X2.T).astype(np.float32)

    @staticmethod
    def polynomial(X1, X2, degree=3, coef0=1.0):
        return ((X1 @ X2.T + coef0) ** degree).astype(np.float32)

    @staticmethod
    def rbf(X1, X2, gamma=0.01):
        x1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
        x2_sq = np.sum(X2 ** 2, axis=1, keepdims=True).T
        dist = x1_sq + x2_sq - 2.0 * (X1 @ X2.T)
        dist = np.maximum(dist, 0.0)
        return np.exp(-gamma * dist).astype(np.float32)


class BinaryKernelSVM:
    """
    Binary kernel SVM trained from scratch using accelerated projected gradient ascent.
    """

    def __init__(
        self,
        kernel="rbf",
        C=1.0,
        gamma=0.01,
        degree=3,
        coef0=1.0,
        lr=0.001,
        epochs=1000,
        tol=1e-6
    ):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.lr = lr
        self.epochs = epochs
        self.tol = tol

        self.support_vectors = None
        self.support_labels = None
        self.support_alpha = None
        self.b = 0.0

    def _kernel(self, X1, X2):
        if self.kernel == "linear":
            return KernelFunctions.linear(X1, X2)
        elif self.kernel == "poly":
            return KernelFunctions.polynomial(X1, X2, self.degree, self.coef0)
        elif self.kernel == "rbf":
            return KernelFunctions.rbf(X1, X2, self.gamma)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

    def fit(self, X, y, K_precomputed=None):
        """Train using Nesterov-accelerated projected gradient ascent on the dual."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        n = X.shape[0]

        # Per-sample C to handle class imbalance in OvA
        n_pos = float(np.sum(y == 1.0))
        n_neg = float(np.sum(y == -1.0))
        C_pos = self.C * (n_neg / n_pos) if n_pos > 0 else self.C
        C_vec = np.where(y == 1.0, C_pos, self.C).astype(np.float32)

        K = K_precomputed if K_precomputed is not None else self._kernel(X, X)
        Q = (y[:, None] * y[None, :]) * K

        # Estimate spectral norm for optimal step size
        rng = np.random.default_rng(42)
        v = rng.standard_normal(n).astype(np.float32)
        for _ in range(15):
            v = Q @ v
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
        L = float(np.linalg.norm(Q @ v))
        step = 1.8 / L  # aggressive but stable step size

        alpha = np.zeros(n, dtype=np.float32)
        alpha_prev = np.zeros(n, dtype=np.float32)
        prev_dual = -np.inf

        for epoch in range(self.epochs):
            # Nesterov momentum
            t = epoch + 1
            momentum = (t - 1.0) / (t + 2.0)
            z = alpha + momentum * (alpha - alpha_prev)
            np.clip(z, 0.0, C_vec, out=z)

            grad = 1.0 - Q @ z
            alpha_new = z + step * grad

            # Project: clip to box then fix equality constraint (alternating projection)
            np.clip(alpha_new, 0.0, C_vec, out=alpha_new)
            for _ in range(5):
                corr = (alpha_new @ y) / (y @ y)
                alpha_new -= corr * y
                np.clip(alpha_new, 0.0, C_vec, out=alpha_new)

            alpha_prev = alpha
            alpha = alpha_new

            # Check convergence
            if (epoch + 1) % 50 == 0:
                dual_obj = float(np.sum(alpha) - 0.5 * alpha @ (Q @ alpha))
                if abs(dual_obj - prev_dual) < self.tol:
                    break
                prev_dual = dual_obj

        # Extract support vectors
        support = alpha > 1e-5
        if not np.any(support):
            raise ValueError("No support vectors found.")

        self.support_vectors = X[support].astype(np.float32)
        self.support_labels = y[support].astype(np.float32)
        self.support_alpha = alpha[support].astype(np.float32)

        # Compute bias from margin support vectors
        margin_sv = (alpha > 1e-5) & (alpha < C_vec - 1e-5)
        if np.any(margin_sv):
            sv_idx = np.where(margin_sv)[0]
        else:
            sv_idx = np.where(support)[0]

        b_vals = []
        for i in sv_idx[:min(100, len(sv_idx))]:
            decision = float(np.sum(alpha * y * K[:, i]))
            b_vals.append(float(y[i] - decision))
        self.b = float(np.median(b_vals))

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        K = self._kernel(X, self.support_vectors).astype(np.float32)
        return K @ (self.support_alpha * self.support_labels) + self.b

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)


class OneVsAllKernelSVM:
    def __init__(self, **kwargs):
        self.params = kwargs
        self.models = {}
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.models = {}

        # Precompute kernel matrix ONCE (shared across all 8 binary classifiers)
        tmp = BinaryKernelSVM(**self.params)
        K = tmp._kernel(X, X).astype(np.float32)

        for c in self.classes:
            y_binary = np.where(y == c, 1.0, -1.0)
            model = BinaryKernelSVM(**self.params)
            model.fit(X, y_binary, K_precomputed=K)
            self.models[c] = model

    def decision_function(self, X):
        scores = []
        for c in self.classes:
            s = self.models[c].decision_function(X).ravel()
            scores.append(s)
        return np.column_stack(scores)

    def predict(self, X):
        scores = self.decision_function(X)
        idx = np.argmax(scores, axis=1)
        return self.classes[idx]


def accuracy_score(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def confusion_matrix_manual(y_true, y_pred, labels):
    label_map = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)

    for yt, yp in zip(y_true, y_pred):
        cm[label_map[yt], label_map[yp]] += 1

    return cm


def load_split(split_name):
    df = pd.read_csv(ANNOTATION_FILE)
    df = df[df["split"] == split_name].copy()

    img_dir = IMAGES_DIR / split_name

    X = np.zeros((len(df), 96 * 96), dtype=np.float64)
    y = df["label"].astype(int).to_numpy()
    filenames = df["file_name"].tolist()

    for i, filename in enumerate(tqdm(filenames, desc=f"Loading {split_name}")):
        img = Image.open(img_dir / filename).convert("L")
        X[i] = np.array(img, dtype=np.float64).flatten()

    return X, y


def subsample(X, y, max_per_class, seed=42):
    rng = np.random.default_rng(seed)
    keep = []

    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, max_per_class, replace=False)
        keep.extend(idx.tolist())

    keep = np.array(sorted(keep))
    return X[keep], y[keep]


def hyperparameter_search(X_train, y_train, X_val, y_val):
    search_space = [
        # Linear
        {"kernel": "linear", "C": 10.0, "epochs": 1000},
        {"kernel": "linear", "C": 100.0, "epochs": 1000},
        # RBF
        {"kernel": "rbf", "C": 10.0, "gamma": 0.01, "epochs": 1000},
        {"kernel": "rbf", "C": 10.0, "gamma": 0.02, "epochs": 1000},
        {"kernel": "rbf", "C": 50.0, "gamma": 0.01, "epochs": 1000},
        {"kernel": "rbf", "C": 50.0, "gamma": 0.02, "epochs": 1000},
        {"kernel": "rbf", "C": 100.0, "gamma": 0.01, "epochs": 1000},
        # Polynomial
        {"kernel": "poly", "C": 10.0, "degree": 3, "coef0": 1.0, "epochs": 1000},
        {"kernel": "poly", "C": 10.0, "degree": 4, "coef0": 1.0, "epochs": 1000},
        {"kernel": "poly", "C": 50.0, "degree": 3, "coef0": 1.0, "epochs": 1000},
        {"kernel": "poly", "C": 50.0, "degree": 4, "coef0": 1.0, "epochs": 1000},
        {"kernel": "poly", "C": 100.0, "degree": 3, "coef0": 1.0, "epochs": 1000},
        {"kernel": "poly", "C": 100.0, "degree": 4, "coef0": 1.0, "epochs": 1000},
    ]

    best_model = None
    best_params = None
    best_val_acc = -1.0

    for i, params in enumerate(search_space, start=1):
        print(f"\n[{i}/{len(search_space)}] {params}")

        try:
            model = OneVsAllKernelSVM(**params)
            model.fit(X_train, y_train)
        except ValueError as e:
            print(f"  Skipped: {e}")
            continue

        val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        print(f"  Val accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = params
            best_model = model

    return best_model, best_params, best_val_acc


def main():
    print("=== SVM 4 (Nesterov accelerated + kernel caching) ===")
    print("Loading data...")

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    X_train, y_train = subsample(X_train, y_train, 200, seed=42)
    X_val, y_val = subsample(X_val, y_val, 100, seed=43)
    X_test, y_test = subsample(X_test, y_test, 100, seed=44)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    with open(PCA_CACHE_FILE, "rb") as f:
        pca = pickle.load(f)

    n_components = 150
    print(f"Applying PCA ({n_components} components)...")

    Z_train = pca.apply_projection(X_train, n_components)
    Z_val = pca.apply_projection(X_val, n_components)
    Z_test = pca.apply_projection(X_test, n_components)

    # Normalize features (zero mean, unit variance)
    z_mean = Z_train.mean(axis=0)
    z_std = Z_train.std(axis=0)
    z_std[z_std == 0] = 1.0
    Z_train = (Z_train - z_mean) / z_std
    Z_val = (Z_val - z_mean) / z_std
    Z_test = (Z_test - z_mean) / z_std

    best_model, best_params, best_val_acc = hyperparameter_search(
        Z_train, y_train, Z_val, y_val
    )

    val_pred = best_model.predict(Z_val)
    test_pred = best_model.predict(Z_test)

    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)

    labels = np.unique(np.concatenate([y_train, y_val, y_test]))
    cm = confusion_matrix_manual(y_test, test_pred, labels)

    print("\nFinal Results")
    print("-" * 80)
    print("Best hyperparameters:", best_params)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print("\nConfusion Matrix")
    print(cm)


if __name__ == "__main__":
    main()
