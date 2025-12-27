import pandas as pd
from quantitative_statistics import mean, std_dev


def run():
    df = pd.read_csv("Dataset/cleaned_full_data.csv")
    quant_features = ['age', 'avg_glucose_level', 'bmi']

    df_quant = df[quant_features]
    df_std = df_quant.copy()

    for feature in quant_features:
        data = df_quant[feature].dropna().tolist()
        mu = mean(data)
        sigma = std_dev(data)
        df_std[feature] = (df_quant[feature] - mu) / sigma

    df_std.to_csv("Dataset/standardized_quantitative_only.csv", index=False)
    print("Descriptive standardization completed.")


if __name__ == "__main__":
    run()
