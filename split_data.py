import pandas as pd
from sklearn.model_selection import train_test_split

# =====================================================
# 1. LOAD CLEANED DATASET
# =====================================================
df = pd.read_csv("Dataset/cleaned_full_data.csv")

# =====================================================
# 2. DEFINE FEATURES & TARGET
# =====================================================
quant_features = ['age', 'avg_glucose_level', 'bmi',
                  'hypertension', 'heart_disease']

X = df[quant_features]
y = df['stroke']

# =====================================================
# 3. STRATIFIED TRAIN–TEST SPLIT (80% / 20%)
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y          # 🔥 THIS IS THE KEY LINE
)

# =====================================================
# 4. SAVE SPLITS
# =====================================================
X_train.to_csv("Dataset/X_train.csv", index=False)
X_test.to_csv("Dataset/X_test.csv", index=False)
y_train.to_csv("Dataset/y_train.csv", index=False)
y_test.to_csv("Dataset/y_test.csv", index=False)

# =====================================================
# 5. VERIFY CLASS DISTRIBUTION
# =====================================================
print("Training class distribution:")
print(y_train.value_counts(normalize=True))

print("\nTesting class distribution:")
print(y_test.value_counts(normalize=True))
print("Total samples:", len(X_train) + len(X_test))
print("Training %:", len(X_train) / (len(X_train) + len(X_test)))
print("Testing %:", len(X_test) / (len(X_train) + len(X_test)))