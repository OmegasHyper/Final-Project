import pandas as pd
import numpy as np

# =====================================================
# Load cleaned dataset
# =====================================================
df = pd.read_csv("Dataset/cleaned_full_data.csv")

# Continuous quantitative features ONLY
quant_features = ['age', 'avg_glucose_level', 'bmi']

df_quant = df[quant_features]

# =====================================================
# Reuse descriptive-statistics functions
# =====================================================
def mean(x):
    return sum(x) / len(x)

def variance(x):
    mu = mean(x)
    return sum((xi - mu) ** 2 for xi in x) / (len(x) - 1)

def std_dev(x):
    return variance(x) ** 0.5

# =====================================================
# Compute statistics using YOUR calculations
# =====================================================
means = {}
stds = {}

for feature in quant_features:
    data = df_quant[feature].dropna().tolist()
    means[feature] = mean(data)
    stds[feature] = std_dev(data)

# =====================================================
# Standardize using YOUR statistics
# =====================================================
df_quant_std = df_quant.copy()

for feature in quant_features:
    df_quant_std[feature] = (
        df_quant[feature] - means[feature]
    ) / stds[feature]

# =====================================================
# Save ONLY standardized quantitative features
# =====================================================
df_quant_std.to_csv(
    "Dataset/standardized_quantitative_only.csv",
    index=False
)

print("Standardization completed using descriptive-statistics calculations.")