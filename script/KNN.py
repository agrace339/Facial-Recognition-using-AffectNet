import numpy as np

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
		dists = np.sqrt(np.sum((new_faces[:, np.newaxis] - self.X_training) ** 2, axis=2))
		
		knn_indices = np.argsort(dists, axis=1)[:, : self.num_of_neighbors]
		knn_labels = self.y_training[knn_indices]

		classes = np.unique(self.y_training)
		probs = []
		for i in range(len(new_faces)):
			counts = [np.sum(knn_labels[i] == c) for c in classes]
			probs.append(np.array(counts) / self.num_of_neighbors)
		return np.array(probs), classes