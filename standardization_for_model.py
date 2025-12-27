import pandas as pd

# =====================================================
# 1. LOAD TRAIN AND TEST SPLITS
# =====================================================
X_train = pd.read_csv("Dataset/X_train.csv")
X_test  = pd.read_csv("Dataset/X_test.csv")

# =====================================================
# 2. DEFINE FEATURE TYPES
# =====================================================
# Continuous quantitative features
quant_features = ['age', 'avg_glucose_level', 'bmi']

# Binary features (do NOT standardize)
binary_features = ['hypertension', 'heart_disease']

# =====================================================
# 3. COMPUTE MEAN & STD FROM TRAINING DATA ONLY
# =====================================================
means = {}
stds = {}

for feature in quant_features:
    mu = X_train[feature].mean()
    sigma = X_train[feature].std(ddof=1)

    means[feature] = mu
    stds[feature] = sigma

# =====================================================
# 4. STANDARDIZE TRAINING DATA
# =====================================================
X_train_std = X_train.copy()

for feature in quant_features:
    X_train_std[feature] = (
        X_train[feature] - means[feature]
    ) / stds[feature]

# =====================================================
# 5. STANDARDIZE TEST DATA (USING TRAIN STATS)
# =====================================================
X_test_std = X_test.copy()

for feature in quant_features:
    X_test_std[feature] = (
        X_test[feature] - means[feature]
    ) / stds[feature]

# =====================================================
# 6. SAVE STANDARDIZED DATASETS
# =====================================================
X_train_std.to_csv("Dataset/X_train_std.csv", index=False)
X_test_std.to_csv("Dataset/X_test_std.csv", index=False)

# =====================================================
# 7. SANITY CHECK (OPTIONAL)
# =====================================================
print("Standardization for model completed.\n")
print("Training data mean (should be ~0 for quantitative features):")
print(X_train_std[quant_features].mean())

print("\nTraining data std (should be ~1 for quantitative features):")
print(X_train_std[quant_features].std(ddof=1))
