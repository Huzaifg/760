import numpy as np

# Implement a logistic regression classifier


class LogisticRegressionClassifier:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.W = None
        self.probabilities = None

    def fit(self, X, y, lr=0.01, epochs=1000):
        """
        Fit the classifier to the training data.

        Parameters:
        - X: A 2D numpy array where each row represents a training sample.
        - y: A 1D numpy array containing the corresponding class labels for each training sample.
        """
        # Add a column of 1s to X
        # X = np.column_stack((np.ones(X.shape[0]), X))
        self.X_train = X
        self.y_train = y

        # Initialize weights to zero
        self.W = np.zeros(X.shape[1])
        # Perform gradient descent
        for i in range(epochs):
            sigmoid = self.sigmoid(np.dot(self.X_train, self.W))
            gradient = np.dot(self.X_train.T, (sigmoid - self.y_train))
            self.W -= lr * gradient

    def predict(self, X_test):
        """
        Predict the class labels for a set of test samples.

        Parameters:
        - X_test: A 2D numpy array where each row represents a test sample.

        Returns:
        - y_pred: A 1D numpy array containing the predicted class labels for each test sample.
        """
        # Calculate probability of each sample
        self.probabilities = self.sigmoid(np.dot(X_test, self.W))

        # Classify samples based on probability
        y_pred = np.where(self.probabilities >= 0.5, 1, 0)
        return y_pred

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict_proba(self):
        return self.probabilities
