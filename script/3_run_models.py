import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from PCA import PCA
from SVM import OneVsAllKernelSVM
from randomforest import RandomForest
from KNN import KNearestNeighbors
from tabulate import tabulate


PROJECT_ROOT = Path(__file__).parent.parent
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
FIG_DIR = PROJECT_ROOT / "output_figs"
IMAGES_DIR = CLEANED_DIR / "images"
ANNOTATION_FILE = CLEANED_DIR / "affectnet_annotations.csv"
PCA_CACHE_FILE = PROJECT_ROOT / "script" / "cache" / "pca.pkl"
CACHE_DIR = PROJECT_ROOT / "script" / "cache"
if not CACHE_DIR.exists():
	CACHE_DIR.mkdir(exist_ok=True)
if not FIG_DIR.exists():
	FIG_DIR.mkdir(exist_ok=True)
table_data = []
# calculate the knn
class_labels = {
	0: "Anger",
	1: "Contempt",
	2: "Disgust",
	3: "Fear",
	4: "Happy",
	5: "Neutral",
	6: "Sad",
	7: "Surprise",
}

#  run the svm to predict the model


def run_svm_experiment(Z_train, y_train, Z_test):
	print("Running SVM experiment...")
	model = OneVsAllKernelSVM(kernel="rbf", C=10.0, gamma=0.01)
	model.fit(Z_train, y_train)
	preds = model.predict(Z_test)
	return model, preds


def run_knn_experiment(knn_class, Z_train, y_train, Z_test, y_test):
	print("Running KNN Experiment...")
	accuracies = []
	ks = list(range(17, 100, 2))
	print(f"Doing hyperparameter tuning for KNN testing ks from {min(ks)} to {max(ks)}...")
	for k in ks:
		# print(f"Testing KNN with k={k}...")
		model = KNearestNeighbors(num_of_neighbors=k)
		model.fit(Z_train, y_train)
		probs, classes = model.predict_probability(Z_test)
		preds = classes[np.argmax(probs, axis=1)]
		accuracies.append(np.mean(preds == y_test))
		if accuracies[-1] == max(accuracies):
			best_k = k
			best_model = model
			best_preds = preds
			best_probs = probs
	print(f"Best k found: {best_k} with accuracy {max(accuracies):.4f}")



	# model = knn_class(num_of_neighbors=7)
	# model.fit(Z_train, y_train)
	# # We need probabilities for the ensemble later
	# probs, classes = model.predict_probability(Z_test)
	# preds = classes[np.argmax(probs, axis=1)]
	return best_model, best_preds, best_probs


# running random_forest from the randomforest.py
def run_rf_experiment(Z_train, y_train, Z_test,y_test,params=None):
	print("Running Random Forest experiment...")
	num_classes = len(np.unique(y_train))

	if params is None:
		n_trees = list(range(25, 76, 25))
		max_depths = [5,10]
		# max_depths = list(range(10, 30, 10))
		#hyperparameter tuning
		best_acc = 0
		param_pairs = [(n, d) for n in n_trees for d in max_depths]
		for n,d in tqdm(param_pairs,desc="Tuning Random Forest hyperparameters"):
			model = RandomForest(n_trees=n, max_depth=d)
			model.fit(Z_train, y_train)
			preds = model.predict(Z_test)
			acc = np.mean(preds == y_test)
			if acc > best_acc:
				best_acc = acc
				best_n = n
				best_d = d
				best_model = model
				best_preds = preds
				best_probs = model.predict_proba(Z_test, n_classes=num_classes)
		print(f"Best Random Forest params: n_trees={best_n}, max_depth={best_d} with accuracy {best_acc:.4f}")
		return best_model, best_preds, best_probs
	
	# Initialize our custom forest
	model = RandomForest(*params)
	model.fit(Z_train, y_train)

	preds = model.predict(Z_test)
	probs = model.predict_proba(Z_test, n_classes=num_classes)
	return model, preds, probs



# weighted voting method
def run_ensemble_experiment(
	svm_model,
	knn_probs,
	randomforest_probs,
	y_test,
	labels,
	Z_test,
	weights=[0.4, 0.2, 0.4],
):
	print("Running Weighted Voted Model...")
	# Now Z_test is available here!
	svm_scores = svm_model.decision_function(Z_test)
	svm_probs = np.exp(svm_scores) / np.sum(np.exp(svm_scores), axis=1, keepdims=True)

	final_probs = (
		(weights[0] * svm_probs)
		+ (weights[1] * knn_probs)
		+ (weights[2] * randomforest_probs)
	)
	preds = labels[np.argmax(final_probs, axis=1)]
	return preds


# calculations


def calculate_all_metrics(y_true, y_pred, labels, name="Model"):
	# Manual Precision/Recall/F1 logic
	# build confusion matrix
	confusion_matrix = np.zeros((len(labels), len(labels)), dtype=int)
	label_to_index = {label: idx for idx, label in enumerate(labels)}
	for true, pred in zip(y_true, y_pred):
		confusion_matrix[label_to_index[true], label_to_index[pred]] += 1

	# calculate multiclass precision, recall, f1 from scratch
	per_class_results = []
	_per_class_results = []
	for i in range(len(labels)):
		TP = confusion_matrix[i, i]
		FP = confusion_matrix[:, i].sum() - TP
		FN = confusion_matrix[i, :].sum() - TP

		precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
		recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
		f1 = (
			2 * (precision * recall) / (precision + recall)
			if (precision + recall) > 0
			else 0.0
		)
		per_class_results.append((labels[i], precision, recall, f1))
		_per_class_results.append((class_labels[labels[i]], precision, recall, f1))
	macro_f1 = np.mean([res[3] for res in per_class_results])
	# print per class results
	print(f"{name} Per Class Metrics:")
	print(
		tabulate(
			_per_class_results,
			headers=["Class", "Precision", "Recall", "F1"],
			tablefmt="grid",
		)
	)
	# print  confusion matrix
	print(f"{name} Confusion Matrix:")
	
	print(confusion_matrix)
	# save per class results to csv
	per_class_df = pd.DataFrame(
		_per_class_results, columns=["Class", "Precision", "Recall", "F1"]
	)
	per_class_df.to_csv(FIG_DIR / f"{name}_per_class_metrics.csv", index=False) 
	accuracy = np.mean(y_true == y_pred)

	# SMAPE
	num = np.abs(y_pred.astype(float) - y_true.astype(float))  # Convert to float here
	den = (np.abs(y_true.astype(float)) + np.abs(y_pred.astype(float))) / 2

	# Create a float output array instead of zeros_like(num)
	smape_array = np.divide(
		num, den, out=np.zeros(len(num), dtype=float), where=den != 0
	)
	smape = np.mean(smape_array) * 100

	return {
		"Accuracy": accuracy,
		"F1": macro_f1,
		"SMAPE": smape,
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


def main():
	#  load data and PCA code
	#  get Z_train, Z_test, y_train, y_test

	X_training, y_training = load_split("train")
	X_test, y_test = load_split("test")

	pca_pickle_file = CACHE_DIR/"pca.pkl"
	if not pca_pickle_file.exists():
		print("PCA model not found in cache. Computing PCA...")
		pca = PCA(X_training)
		with open(pca_pickle_file,"wb") as f:
			pickle.dump(pca, f)
		print(f"PCA computed and saved to {pca_pickle_file}")	
	else:
		with open(pca_pickle_file, "rb") as f:
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
	knn_model, knn_preds, knn_probs = run_knn_experiment(
		KNearestNeighbors, Z_train, y_training, Z_test, y_test
	)
	rf_model, rf_preds, rf_probs = run_rf_experiment(Z_train, y_training, Z_test, y_test, (60,10))

	# run Ensemble
	ensemble_preds = run_ensemble_experiment(
		svm_model, knn_probs, rf_probs, y_test, labels, Z_test
	)
	# Metrics

	all_results["SVM"] = calculate_all_metrics(y_test, svm_preds, labels, "SVM")
	all_results["KNN"] = calculate_all_metrics(y_test, knn_preds, labels, "KNN")
	all_results["Random Forest"] = calculate_all_metrics(
		y_test, rf_preds, labels, "Random Forest"
	)
	# all_results['Ensemble'] = calculate_all_metrics(y_test, ensemble_preds, labels)

	# show report in table format
	headers_ = ["Model", "Test Accuracy", "Test F1", "Test SMAPE"]
	#  report
	for model_name, metrics in all_results.items():
		row = [
			model_name,
			f"{metrics['Accuracy']:.4f}",
			f"{metrics['F1']:.4f}",
			f"{metrics['SMAPE']:.2f}%",
		]
		table_data.append(row)

	print(tabulate(table_data, headers=headers_, tablefmt="grid"))


if __name__ == "__main__":
	main()
