import pandas as pd
from sklearn.model_selection import train_test_split


def run():
    df = pd.read_csv("Dataset/cleaned_full_data.csv")

    features = ['age', 'avg_glucose_level', 'bmi',
                'hypertension', 'heart_disease']

    X = df[features]
    y = df['stroke']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_train.to_csv("Dataset/X_train.csv", index=False)
    X_test.to_csv("Dataset/X_test.csv", index=False)
    y_train.to_csv("Dataset/y_train.csv", index=False)
    y_test.to_csv("Dataset/y_test.csv", index=False)

    print("Data split completed (80/20, stratified).")


if __name__ == "__main__":
    run()
