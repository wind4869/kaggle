import numpy as np


class LogisticRegression:
    def __init__(self, alpha=1, max_iter=100, threshold=0.5):
        self._alpha = alpha
        self._max_iter = max_iter
        self._threshold = threshold
        self._weights = None

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    def fit(self, X, y):
        y_mat = np.mat(y).transpose()
        number, dimension = np.shape(X)
        X_mat = np.mat(np.column_stack((X, np.ones(number))))
        self._weights = np.mat(np.ones((dimension + 1, 1)))

        for i in range(self._max_iter):
            error = y_mat - self.sigmoid(X_mat * self._weights)
            self._weights += self._alpha * (X_mat.transpose() * error)

    def do_predict(self, x):
        return np.ceil(self.sigmoid(x) - self._threshold)

    def predict(self, X):
        number, _ = np.shape(X)
        X_mat = np.mat(np.column_stack((X, np.ones(number))))
        y_mat = self.do_predict(X_mat * self._weights)
        return np.array(y_mat).astype(np.int32).reshape(-1, )
