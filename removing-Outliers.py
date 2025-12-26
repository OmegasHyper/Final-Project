import pandas as pd
from scipy.stats import zscore

df = pd.read_csv("Dataset/full_data.csv")
num_cols = ['age', 'avg_glucose_level', 'bmi']

z_scores = df[num_cols].apply(zscore)

threshold = 3

df_clean = df[(z_scores.abs() <= threshold).all(axis=1)]

df_clean.to_csv("Dataset/cleaned_full_data.csv", index=False)