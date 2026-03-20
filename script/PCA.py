# %%
"""
PCA Analysis for Facial Recognition

This script performs Principal Component Analysis (PCA) on facial images from the AffectNet dataset.
It includes a PCA class for dimensionality reduction and reconstruction, and a main function that
loads training images, computes PCA, visualizes the top components, and creates a reconstruction animation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import pickle
from tqdm import tqdm

# %%

PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_DIR = PROJECT_ROOT / "data" / "cleaned" / "images" / "train"
CACHE_DIR = PROJECT_ROOT / "script" / "cache"
if not CACHE_DIR.exists():
	CACHE_DIR.mkdir(exist_ok=True)

class PCA:
	"""
	Principal Component Analysis class for dimensionality reduction.

	This class standardizes the input data, computes the principal components via SVD,
	and provides methods for projecting data onto lower dimensions and reconstructing it.
	"""
	def __init__(self, X_train: np.ndarray):
		"""
		Initialize the PCA object with training data.

		Args:
			X_train (np.ndarray): Training data matrix of shape (n_samples, n_features).

		Raises:
			ValueError: If X_train is not a 2D array or has zero samples/features.
		"""
		if X_train.ndim != 2:
			raise ValueError("X_train must be a 2D array")
		if X_train.shape[0] == 0 or X_train.shape[1] == 0:
			raise ValueError("X_train must have at least one sample and one feature")
		# Standardize Data
		self.train_mean = np.mean(X_train, axis=0)
		self.train_dev = np.std(X_train, ddof=1, axis=0)

		X = (X_train - self.train_mean) / self.train_dev

		# PCA via SVD
		U, s, Vt = np.linalg.svd(X, full_matrices=False)
		self.principal_components = Vt.T

	def apply_projection(self, X: np.ndarray, n_components: int = 2):
		"""
		Project data onto the first n principal components.

		Args:
			X (np.ndarray): Data to project.
			n_components (int): Number of components to use. Defaults to 2.

		Returns:
			np.ndarray: Projected data of shape (n_samples, n_components).

		Raises:
			ValueError: If n_components is invalid.
		"""
		if n_components < 1 or n_components > self.principal_components.shape[1]:
			raise ValueError(f"n_components must be between 1 and {self.principal_components.shape[1]}")
		_X = (X - self.train_mean) / self.train_dev
		return _X @ self.principal_components[:, :n_components]

	def reconstruct_data(self, Z: np.ndarray, n_components: int = 2):
		"""
		Reconstruct data from projected coordinates.

		Args:
			Z (np.ndarray): Projected data of shape (n_samples, n_components).
			n_components (int): Number of components used in projection. Defaults to 2.

		Returns:
			np.ndarray: Reconstructed data of original shape.

		Raises:
			ValueError: If n_components is invalid.
		"""
		if n_components < 1 or n_components > self.principal_components.shape[1]:
			raise ValueError(f"n_components must be between 1 and {self.principal_components.shape[1]}")
		reconstructed = Z @ self.principal_components[:,:n_components].T
		return (reconstructed * self.train_dev) + self.train_mean
		
		


# %%
def main():
	"""
	Main function to perform PCA analysis on facial images.

	Loads training images, computes PCA, saves the model, plots top 2 components,
	and creates a reconstruction animation for a sample image.
	"""
	train_files = list(TRAIN_DIR.glob("*.png"))
	forder = []
	X_train = np.ndarray((len(train_files), 96**2))
	for i, f in enumerate(tqdm(train_files, desc="Loading train images")):
		X_train[i, :] = np.array(Image.open(f)).flatten()
		forder.append(f.name)
	pca_pickle_file = CACHE_DIR/"pca.pkl"
	if pca_pickle_file.exists():
		with open(pca_pickle_file, "rb") as f:
			pca = pickle.load(f)
		print(f"Loaded PCA object from {pca_pickle_file}")
	else:
		print(f"No PCA object found in cache. Recalculating... (Takes a few minutes)")
		pca = PCA(X_train)
		with open(pca_pickle_file,"wb") as f:
			pickle.dump(pca, f)
		print(f"Completed PCA and saved object copy to {pca_pickle_file}")
	# %%
	OUTPUT_FIG_DIR = PROJECT_ROOT / "output_figs"
	OUTPUT_FIG_DIR.mkdir(exist_ok=True)

	X = (X_train - pca.train_mean) / pca.train_dev
	top2 = pca.principal_components[:, :2]
	projected = X @ top2
	plt.scatter(projected[:, 0], projected[:, 1])
	plt.title("PCA 2 most significant components")
	plt.tight_layout()
	plt.savefig(OUTPUT_FIG_DIR/"pca_top_2_components.png")
	# %%
	reconstruct_file_idx = 3180
	outfile = OUTPUT_FIG_DIR/f"pca_reconstruction_{forder[reconstruct_file_idx][:-4]}.mp4"
	im = X_train[reconstruct_file_idx,:]
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	out = cv2.VideoWriter(outfile, fourcc, 60.0, (96,96), 0)

	print(f"Making reconstruction animation and saving to {outfile}")
	for idx in tqdm(range(1, pca.principal_components.shape[1], 10), desc="Creating reconstruction video"):
		z = pca.apply_projection(im, n_components=idx)
		reconstructed = pca.reconstruct_data(z, n_components=idx)
		frame = np.clip(reconstructed, 0, 255).astype(np.uint8)
		out.write(np.reshape(frame,(96,96)))
	out.release()
#%%
if __name__ == "__main__":
	main()
#%%