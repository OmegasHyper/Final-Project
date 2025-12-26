import pandas as pd
import numpy as np

# Load cleaned dataset
df = pd.read_csv("Dataset/cleaned_full_data.csv")

# Continuous quantitative features
quant_features = ['age', 'avg_glucose_level', 'bmi']

# ===== Manual statistical functions =====
def mean(x):
    return sum(x) / len(x)

def variance(x):
    mu = mean(x)
    return sum((xi - mu) ** 2 for xi in x) / (len(x) - 1)

def std_dev(x):
    return variance(x) ** 0.5

def median(x):
    x_sorted = sorted(x)
    n = len(x)
    mid = n // 2
    if n % 2 == 0:
        return (x_sorted[mid - 1] + x_sorted[mid]) / 2
    else:
        return x_sorted[mid]

def mode(x):
    values, counts = np.unique(x, return_counts=True)
    return values[np.argmax(counts)]

def data_range(x):
    return max(x) - min(x)

# ===== Compute statistics =====
stats = []

for feature in quant_features:
    data = df[feature].dropna().tolist()

    stats.append({
        "Feature": feature,
        "Mean": mean(data),
        "Median": median(data),
        "Mode": mode(data),
        "Variance": variance(data),
        "Std_Deviation": std_dev(data),
        "Range": data_range(data)
    })

stats_df = pd.DataFrame(stats)

# Display and save
print(stats_df)
stats_df.to_csv("Dataset/quantitative_statistics.csv", index=False)
