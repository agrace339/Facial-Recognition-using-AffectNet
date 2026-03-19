import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from PCA_3 import PCA
from SVM_4 import OneVsAllKernelSVM
from randomforest import CustomRandomForest
from tabulate import tabulate


PROJECT_ROOT = Path(__file__).parent.parent
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
IMAGES_DIR = CLEANED_DIR / "images"
ANNOTATION_FILE = CLEANED_DIR / "affectnet_annotations.csv"
PCA_CACHE_FILE = PROJECT_ROOT / "script" / "cache" / "pca.pkl"


table_data = []
# calculate the knn

class KNearestNeighbors:
    def __init__(self, num_of_neighbors=5):
        self.num_of_neighbors = num_of_neighbors
        self.X_training = None
        self.y_training = None

    def fit(self, pixels_training_set, num_code_emotion):
        self.X_training = np.asarray(pixels_training_set, dtype=np.float32)
        self.y_training = np.asarray(num_code_emotion)

    def predict_probability(self, new_faces):
        """Calculates confidence based on neighbor frequency."""
        new_faces = np.asarray(new_faces, dtype=np.float32)
        # Vectorized Euclidean Distance
        dists = np.sqrt(np.sum(new_faces ** 2, axis=1, keepdims=True) +
                        np.sum(self.X_training ** 2, axis=1) -
                        2.0 * (new_faces @ self.X_training.T))
        knn_indices = np.argsort(dists, axis=1)[:, :self.num_of_neighbors]
        knn_labels = self.y_training[knn_indices]

        classes = np.unique(self.y_training)
        probs = []
        for i in range(len(new_faces)):
            counts = [np.sum(knn_labels[i] == c) for c in classes]
            probs.append(np.array(counts) / self.num_of_neighbors)
        return np.array(probs), classes


#  run the svm to predict the model

def run_svm_experiment(Z_train, y_train, Z_test):
    print("svm, predict the model")
    model = OneVsAllKernelSVM(kernel="rbf", C=10.0, gamma=0.01)
    model.fit(Z_train, y_train)
    preds = model.predict(Z_test)
    return model, preds


def run_knn_experiment(knn_class, Z_train, y_train, Z_test):
    print("Running KNN Experiment...")
    model = knn_class(num_of_neighbors=7)
    model.fit(Z_train, y_train)
    # We need probabilities for the ensemble later
    probs, classes = model.predict_probability(Z_test)
    preds = classes[np.argmax(probs, axis=1)]
    return model, preds, probs

# running random_forest from the randomforest.py
def run_rf_experiment(Z_train, y_train, Z_test):
    print("Running custom Random Forest from scratch...")
    num_classes = len(np.unique(y_train))

    # Initialize our custom forest
    model = CustomRandomForest(n_trees=20, max_depth=10)
    model.fit(Z_train, y_train)

    preds = model.predict(Z_test)
    probs = model.predict_proba(Z_test, n_classes=num_classes)

    return model, preds, probs

# esemble method

def run_ensemble_experiment(svm_model, knn_probs, randomforest_probs, y_test, labels, Z_test, weights=[0.4, 0.2, 0.4]):
    print("Running Weighted Ensemble...")
    # Now Z_test is available here!
    svm_scores = svm_model.decision_function(Z_test)
    svm_probs = np.exp(svm_scores) / np.sum(np.exp(svm_scores), axis=1, keepdims=True)

    final_probs = (weights[0] * svm_probs) + (weights[1] * knn_probs) + (weights[2] * randomforest_probs)
    preds = labels[np.argmax(final_probs, axis=1)]
    return preds

# calculations

def calculate_all_metrics(y_true, y_pred, labels):
    # Manual Precision/Recall/F1 logic
    cm, per_class_results, macro_f1 = calculate_metrics(y_true, y_pred, labels)

    accuracy = np.mean(y_true == y_pred)

    # SMAPE
    num = np.abs(y_pred.astype(float) - y_true.astype(float))  # Convert to float here
    den = (np.abs(y_true.astype(float)) + np.abs(y_pred.astype(float))) / 2

    # Create a float output array instead of zeros_like(num)
    smape_array = np.divide(num, den, out=np.zeros(len(num), dtype=float), where=den != 0)
    smape = np.mean(smape_array) * 100

    return {
        "Accuracy": accuracy,
        "F1": macro_f1,
        "SMAPE": smape
    }



def load_split(split_name):
    """Loads image data and labels from the cleaned directory."""
    df = pd.read_csv(ANNOTATION_FILE)
    df = df[df["split"] == split_name].copy()
    img_directory = IMAGES_DIR / split_name

    # Initialize matrix: num_samples x (96*96 pixels)
    matrix_of_pixel = np.zeros((len(df), 96 * 96), dtype=np.float64)
    vector_of_categories = df["label"].astype(int).to_numpy()
    filenames = df["file_name"].tolist()

    for i, filename in enumerate(tqdm(filenames, desc=f"Loading {split_name}")):
        img = Image.open(img_directory / filename).convert("L")
        matrix_of_pixel[i] = np.array(img, dtype=np.float64).flatten()
    return matrix_of_pixel, vector_of_categories


def subsample(features, labels, max_per_class, seed=42):
    """Ensures balanced classes for training."""
    random_number_generator = np.random.default_rng(seed)
    row_numbers_img = []
    for category in np.unique(labels):
        index = np.where(labels == category)[0]
        if len(index) > max_per_class:
            index = random_number_generator.choice(index, max_per_class, replace=False)
        row_numbers_img.extend(index.tolist())
    return features[np.array(sorted(row_numbers_img))], labels[np.array(sorted(row_numbers_img))]


def calculate_metrics(y_true, predicted_y, labels):
    """Manual calculation of Precision, Recall, and F1."""
    label_map = {label: i for i, label in enumerate(labels)}
    confusion_matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for yt, y_predicted in zip(y_true, predicted_y):
        confusion_matrix[label_map[yt], label_map[y_predicted]] += 1

    results = {}
    for i, label in enumerate(labels):
        true_positive = confusion_matrix[i, i]
        false_positive = np.sum(confusion_matrix[:, i]) - true_positive
        false_negative = np.sum(confusion_matrix[i, :]) - true_positive

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        results[label] = (precision, recall, f1_score)

    macro_f1_score = np.mean([v[2] for v in results.values()])
    return confusion_matrix, results, macro_f1_score




def main():
    #  load data and PCA code
    #  get Z_train, Z_test, y_train, y_test

    X_train_raw, y_train_raw = load_split("train")
    X_test_raw, y_test_raw = load_split("test")

    X_training, y_training = subsample(X_train_raw, y_train_raw, 200)
    X_test, y_test = subsample(X_test_raw, y_test_raw, 100)

    if not PCA_CACHE_FILE.exists():
        raise FileNotFoundError(f"PCA cache not found at {PCA_CACHE_FILE}")

    with open(PCA_CACHE_FILE, "rb") as f:
        pca = pickle.load(f)

    Z_train = pca.apply_projection(X_training, n_components=150)
    Z_test = pca.apply_projection(X_test, n_components=150)

    # Standardization
    z_mean, z_std = Z_train.mean(axis=0), Z_train.std(axis=0)
    z_std[z_std == 0] = 1.0
    Z_train = (Z_train - z_mean) / z_std
    Z_test = (Z_test - z_mean) / z_std

    labels = np.unique(y_test)
    all_results = {}

    # run SVM AND KNN independently
    svm_model, svm_preds = run_svm_experiment(Z_train, y_training, Z_test)
    knn_model, knn_preds, knn_probs = run_knn_experiment(KNearestNeighbors, Z_train, y_training, Z_test)
    rf_model, rf_preds, rf_probs = run_rf_experiment(Z_train, y_training, Z_test)

    # run Ensemble
    ensemble_preds = run_ensemble_experiment(svm_model, knn_probs, rf_probs, y_test, labels, Z_test)
    # Metrics
    all_results['SVM'] = calculate_all_metrics(y_test, svm_preds, labels)
    all_results['KNN'] = calculate_all_metrics(y_test, knn_preds, labels)
    all_results['Random Forest'] = calculate_all_metrics(y_test, rf_preds, labels)
    all_results['Ensemble'] = calculate_all_metrics(y_test, ensemble_preds, labels)


    # show report in table format
    headers_ = ["Model","Accuracy","F1","SMAPE"]
    #  report
    for model_name, metrics in all_results.items():
        row = [
            model_name,
            f"{metrics['Accuracy']:.4f}",
            f"{metrics['F1']:.4f}",
            f"{metrics['SMAPE']:.2f}%"

        ]
        table_data.append(row)


    print(tabulate(table_data, headers=headers_, tablefmt="grid"))

if __name__ == "__main__":
    main()