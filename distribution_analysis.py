import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro

# =====================================================
# 1. LOAD TRAINING DATA ONLY
# =====================================================
X_train = pd.read_csv("Dataset/X_train.csv")
y_train = pd.read_csv("Dataset/y_train.csv")

df_train = X_train.copy()
df_train["stroke"] = y_train

quant_features = ['age', 'avg_glucose_level', 'bmi']

# =====================================================
# 2. HISTOGRAMS + NORMALITY TEST
# =====================================================
for feature in quant_features:

    # ----- Histogram -----
    plt.figure()
    plt.hist(df_train[feature], bins=30)
    plt.title(f"Histogram of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.show()

    # ----- Shapiro-Wilk Normality Test -----
    stat, p_value = shapiro(df_train[feature])

    print(f"\nFeature: {feature}")
    print(f"Shapiro-Wilk p-value: {p_value:.5f}")

    if p_value > 0.05:
        print("Conclusion: Approximately Gaussian (fail to reject H0)")
    else:
        print("Conclusion: Not Gaussian (reject H0)")

# =====================================================
# 3. CONDITIONAL DISTRIBUTIONS (P(x | y))
# =====================================================
for feature in quant_features:

    plt.figure()

    for label in [0, 1]:
        subset = df_train[df_train["stroke"] == label]
        plt.hist(
            subset[feature],
            bins=30,
            alpha=0.6,
            label=f"stroke = {label}"
        )

    plt.title(f"Conditional Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
