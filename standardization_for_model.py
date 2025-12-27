import pandas as pd


def run():
    X_train = pd.read_csv("Dataset/X_train.csv")
    X_test = pd.read_csv("Dataset/X_test.csv")

    quant_features = ['age', 'avg_glucose_level', 'bmi']

    means = X_train[quant_features].mean()
    stds = X_train[quant_features].std(ddof=1)

    X_train_std = X_train.copy()
    X_test_std = X_test.copy()

    for f in quant_features:
        X_train_std[f] = (X_train[f] - means[f]) / stds[f]
        X_test_std[f] = (X_test[f] - means[f]) / stds[f]

    X_train_std.to_csv("Dataset/X_train_std.csv", index=False)
    X_test_std.to_csv("Dataset/X_test_std.csv", index=False)

    print("Model standardization completed.")


if __name__ == "__main__":
    run()
