import numpy as np
import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score


# =====================================================
# 1. LOAD STANDARDIZED DATA FOR MODELING
# =====================================================
X_train = pd.read_csv("Dataset/X_train_std.csv")
y_train = pd.read_csv("Dataset/y_train.csv")

X_test = pd.read_csv("Dataset/X_test_std.csv")
y_test = pd.read_csv("Dataset/y_test.csv")

# Convert to NumPy
X_train = X_train.values
y_train = y_train.values.ravel()
X_test = X_test.values
y_test = y_test.values.ravel()


# =====================================================
# 2. GAUSSIAN NAIVE BAYES (FROM SCRATCH)
# =====================================================
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

            # Prior probability P(y=c)
            self.priors[c] = X_c.shape[0] / n_samples

            # Mean and variance per feature
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
            posteriors.append(log_prior + log_likelihood)

        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])


# =====================================================
# 3. TRAIN & PREDICT (FROM SCRATCH)
# =====================================================
gnb = GaussianNaiveBayes()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

# Accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy (from scratch): {accuracy:.4f}")


# =====================================================
# 4. MANUAL CONFUSION MATRIX & METRICS
# =====================================================
TP = np.sum((y_test == 1) & (y_pred == 1))
TN = np.sum((y_test == 0) & (y_pred == 0))
FP = np.sum((y_test == 0) & (y_pred == 1))
FN = np.sum((y_test == 1) & (y_pred == 0))

print("\nConfusion Matrix (From Scratch)")
print(f"TP: {TP}   FP: {FP}")
print(f"FN: {FN}   TN: {TN}")

precision = TP / (TP + FP + 1e-9)
recall = TP / (TP + FN + 1e-9)
f1 = 2 * precision * recall / (precision + recall + 1e-9)

print(f"\nPrecision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")


# =====================================================
# 5. SKLEARN GAUSSIAN NAIVE BAYES (COMPARISON)
# =====================================================
sk_gnb = GaussianNB(var_smoothing=1e-9)
sk_gnb.fit(X_train, y_train)

y_pred_sk = sk_gnb.predict(X_test)

acc_sk = accuracy_score(y_test, y_pred_sk)
prec_sk = precision_score(y_test, y_pred_sk)
rec_sk = recall_score(y_test, y_pred_sk)
f1_sk = f1_score(y_test, y_pred_sk)
cm_sk = confusion_matrix(y_test, y_pred_sk)

print("\n=== sklearn GaussianNB ===")
print(f"Accuracy:  {acc_sk:.4f}")
print(f"Precision: {prec_sk:.4f}")
print(f"Recall:    {rec_sk:.4f}")
print(f"F1-score:  {f1_sk:.4f}")
print("\nConfusion Matrix:")
print(cm_sk)


# =====================================================
# 6. FINAL COMPARISON TABLE
# =====================================================
print("\n=== Comparison ===")
print(f"{'Metric':<10} {'From-scratch':<15} {'sklearn':<15}")
print(f"{'Accuracy':<10} {accuracy:<15.4f} {acc_sk:<15.4f}")
print(f"{'Precision':<10} {precision:<15.4f} {prec_sk:<15.4f}")
print(f"{'Recall':<10} {recall:<15.4f} {rec_sk:<15.4f}")
print(f"{'F1':<10} {f1:<15.4f} {f1_sk:<15.4f}")
