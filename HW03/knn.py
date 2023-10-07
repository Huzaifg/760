import numpy as np


class NearestNeighborClassifier:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.K = 1

    def fit(self, X, y, k):
        """
        Fit the classifier to the training data.

        Parameters:
        - X: A 2D numpy array where each row represents a training sample.
        - y: A 1D numpy array containing the corresponding class labels for each training sample.
        """
        self.X_train = X
        self.y_train = y
        self.K = k
        self.y_probs = []

    def predict(self, X_test):
        """
        Predict the class labels for a set of test samples.

        Parameters:
        - X_test: A 2D numpy array where each row represents a test sample.

        Returns:
        - y_pred: A 1D numpy array containing the predicted class labels for each test sample.
        """
        y_pred = []
        for x in X_test:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            nearest_neighbor_idx = np.argsort(distances)[:self.K]
            # Class is the majority of the nearest neighbors
            nb_label = []
            for idx in nearest_neighbor_idx:
                nb_label.append(int(self.y_train[idx]))

            # Find the most common class
            nb_label = np.array(nb_label)
            nearest_neighbor_class = np.bincount(nb_label).argmax()
            if (len(np.bincount(nb_label))) == 1:
                self.y_probs.append(0)
            else:
                # Find the probabliity of the class being 1
                self.y_probs.append(np.bincount(nb_label)[
                                    1] / np.sum(np.bincount(nb_label)))
            y_pred.append(nearest_neighbor_class)
        return np.array(y_pred)

    def predict_proba(self):
        return self.y_probs
