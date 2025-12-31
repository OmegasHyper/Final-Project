import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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

    def precision_recall_f1(self, y_true, y_pred):
        classes = np.unique(y_true)

        precisions = []
        recalls = []
        f1s = []

        for c in classes:
            tp = np.sum((y_pred == c) & (y_true == c))
            fp = np.sum((y_pred == c) & (y_true != c))
            fn = np.sum((y_pred != c) & (y_true == c))

            precision = tp / (tp + fp + 1e-9)
            recall = tp / (tp + fn + 1e-9)
            f1 = 2 * precision * recall / (precision + recall + 1e-9)

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        accuracy = np.mean(y_true == y_pred)
        # Macro-averaged metrics
        return (
            accuracy,
            np.mean(precisions),
            np.mean(recalls),
            np.mean(f1s),
        )


def run():
    X_train = pd.read_csv("Dataset/X_train_std.csv").values
    X_test = pd.read_csv("Dataset/X_test_std.csv").values
    y_train = pd.read_csv("Dataset/y_train.csv").values.ravel()
    y_test = pd.read_csv("Dataset/y_test.csv").values.ravel()

    nb = GaussianNaiveBayes()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)

    acc = np.mean(y_pred == y_test)

    sk = GaussianNB()
    sk.fit(X_train, y_train)
    y_pred_sk = sk.predict(X_test)
    acc_sk = accuracy_score(y_test, y_pred_sk)

    sk_bernoulli = BernoulliNB()
    sk_bernoulli.fit(X_train, y_train)
    y_pred_sk_bernoulli = sk_bernoulli.predict(X_test)
    acc_sk_bernoulli = accuracy_score(y_test, y_pred_sk_bernoulli)

    acc, p, r, f1 = nb.precision_recall_f1(y_test, y_pred)

    print("Bernoulli sklearn metrics: \n")
    print(f"Accuracy  :   {acc_sk_bernoulli:.15f}")
    print(f"Precision :   {precision_score(y_test, y_pred_sk_bernoulli, average='macro'):.15f}")
    print(f"Recall    :   {recall_score(y_test, y_pred_sk_bernoulli, average='macro'):.15f}")
    print(f"F1-score  :   {f1_score(y_test, y_pred_sk_bernoulli, average='macro'):.15f}")
    
    print('------------------------------------')
    print("sklearn metrics: \n")
    print(f"Accuracy  :   {acc_sk:.15f}")
    print(f"Precision :   {precision_score(y_test, y_pred_sk, average='macro'):.15f}")
    print(f"Recall    :   {recall_score(y_test, y_pred_sk, average='macro'):.15f}")
    print(f"F1-score  :   {f1_score(y_test, y_pred_sk, average='macro'):.15f}")

    print('------------------------------------')
    print("From scratch metrics: \n")
    print(f"Accuracy  :   {acc:.15f}")
    print(f"Precision :   {p:.15f}")
    print(f"Recall    :   {r:.15f}")
    print(f"F1-score  :   {f1:.15f}")


if __name__ == "__main__":
    run()
