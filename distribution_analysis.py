import pandas as pd
from scipy.stats import shapiro
import numpy as np


def run():
    X_train = pd.read_csv("Dataset/X_train.csv")
    y_train = pd.read_csv("Dataset/y_train.csv")

    df = X_train.copy()
    df["stroke"] = y_train

    quant_features = ['age', 'avg_glucose_level', 'bmi']

    for feature in quant_features:
        clean = (
            df[feature]
            .dropna()
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .astype(float)
        )

        stat, p = shapiro(clean.sample(min(3000, len(clean)), random_state=42))
        print(f"{feature} | p-value: {p:.10e}")


if __name__ == "__main__":
    run()
