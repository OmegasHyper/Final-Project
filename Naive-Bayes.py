import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score


# Load training data
X_train = pd.read_csv("Dataset/X_train.csv")
y_train = pd.read_csv("Dataset/y_train.csv")

# Load test data
X_test = pd.read_csv("Dataset/X_test.csv")
y_test = pd.read_csv("Dataset/y_test.csv")

# Convert to numpy
X_train = X_train.values
y_train = y_train.values.ravel()
X_test = X_test.values
y_test = y_test.values.ravel()

class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.priors = {}
        self.means = {}
        self.vars = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples = X.shape[0]

        for c in self.classes:
            X_c = X[y == c]

            # Prior P(y)
            self.priors[c] = X_c.shape[0] / n_samples

            # Mean and variance for each feature
            self.means[c] = np.mean(X_c, axis=0)
            self.vars[c] = np.var(X_c, axis=0) + 1e-9  # numerical stability

    def _log_gaussian(self, x, mean, var):
        return -0.5 * np.sum(
            np.log(2 * np.pi * var) + ((x - mean) ** 2) / var
        )
    
    def _predict_single(self, x):
        posteriors = []

        for c in self.classes:
            log_prior = np.log(self.priors[c])
            log_likelihood = self._log_gaussian(x, self.means[c], self.vars[c])
            posterior = log_prior + log_likelihood
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]
    
    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

    

# Train model
gnb = GaussianNaiveBayes()
gnb.fit(X_train, y_train)

# Predictions
y_pred = gnb.predict(X_test)

# Accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.4f}")


TP = np.sum((y_test == 1) & (y_pred == 1))
TN = np.sum((y_test == 0) & (y_pred == 0))
FP = np.sum((y_test == 0) & (y_pred == 1))
FN = np.sum((y_test == 1) & (y_pred == 0))

print("\nConfusion Matrix")
print(f"TP: {TP}  FP: {FP}")
print(f"FN: {FN}  TN: {TN}")

precision = TP / (TP + FP + 1e-9)
recall = TP / (TP + FN + 1e-9)
f1 = 2 * precision * recall / (precision + recall + 1e-9)

print(f"\nPrecision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")


# Train sklearn Gaussian NB
sk_gnb = GaussianNB(var_smoothing=1e-9)
sk_gnb.fit(X_train, y_train)

# Predict
y_pred_sk = sk_gnb.predict(X_test)

# Metrics
acc_sk = accuracy_score(y_test, y_pred_sk)
prec_sk = precision_score(y_test, y_pred_sk)
rec_sk = recall_score(y_test, y_pred_sk)
f1_sk = f1_score(y_test, y_pred_sk)

cm_sk = confusion_matrix(y_test, y_pred_sk)

print("=== sklearn GaussianNB ===")
print(f"Accuracy:  {acc_sk:.4f}")
print(f"Precision: {prec_sk:.4f}")
print(f"Recall:    {rec_sk:.4f}")
print(f"F1-score:  {f1_sk:.4f}")

print("\nConfusion Matrix:")
print(cm_sk)


print("\n=== Comparison ===")
print(f"{'Metric':<10} {'From-scratch':<15} {'sklearn':<15}")
print(f"{'Accuracy':<10} {accuracy:<15.4f} {acc_sk:<15.4f}")
print(f"{'Precision':<10} {precision:<15.4f} {prec_sk:<15.4f}")
print(f"{'Recall':<10} {recall:<15.4f} {rec_sk:<15.4f}")
print(f"{'F1':<10} {f1:<15.4f} {f1_sk:<15.4f}")
