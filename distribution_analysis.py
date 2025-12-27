import pandas as pd
from scipy.stats import shapiro


def run():
    X_train = pd.read_csv("Dataset/X_train.csv")
    y_train = pd.read_csv("Dataset/y_train.csv")

    df = X_train.copy()
    df["stroke"] = y_train

    quant_features = ['age', 'avg_glucose_level', 'bmi']

    for feature in quant_features:
        stat, p = shapiro(df[feature])
        print(f"{feature} | Shapiro p-value: {p:.5f}")


if __name__ == "__main__":
    run()
