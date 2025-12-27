import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.priors = {}
        self.means = {}
        self.vars = {}

        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = len(X_c) / len(X)
            self.means[c] = np.mean(X_c, axis=0)
            self.vars[c] = np.var(X_c, axis=0) + 1e-9

    def _log_gaussian(self, x, mean, var):
        return -0.5 * np.sum(
            np.log(2 * np.pi * var) + ((x - mean) ** 2) / var
        )

    def predict(self, X):
        preds = []
        for x in X:
            posteriors = []
            for c in self.classes:
                posteriors.append(
                    np.log(self.priors[c]) +
                    self._log_gaussian(x, self.means[c], self.vars[c])
                )
            preds.append(self.classes[np.argmax(posteriors)])
        return np.array(preds)


def run():
    X_train = pd.read_csv("Dataset/X_train_std.csv").values
    X_test = pd.read_csv("Dataset/X_test_std.csv").values
    y_train = pd.read_csv("Dataset/y_train.csv").values.ravel()
    y_test = pd.read_csv("Dataset/y_test.csv").values.ravel()

    nb = GaussianNaiveBayes()
    nb.fit(X_train, y_train)
    acc = np.mean(nb.predict(X_test) == y_test)

    sk = GaussianNB()
    sk.fit(X_train, y_train)
    acc_sk = accuracy_score(y_test, sk.predict(X_test))

    print(f"Accuracy (from scratch): {acc:.4f}")
    print(f"Accuracy (sklearn):      {acc_sk:.4f}")


if __name__ == "__main__":
    run()
