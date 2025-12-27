# 🧠 Naive Bayes Stroke Classification Project

A complete **end-to-end machine learning pipeline** for **binary stroke classification** using a **Gaussian Naive Bayes classifier**, including **data preprocessing, statistical analysis, model implementation from scratch, comparison with scikit-learn, and a fully interactive PyQt5 dashboard**.

---

## 📌 Project Objectives

This project aims to:

- Perform **binary classification** (Stroke / No Stroke)
- Apply **Naive Bayes from scratch**
- Compare results with **scikit-learn’s GaussianNB**
- Conduct **statistical analysis** on quantitative features
- Visualize feature distributions, conditional distributions P(x|y), and standardization effects
- Provide a **professional GUI** that runs the **entire pipeline**

---

## 📊 Dataset

- Source: **Kaggle Stroke Dataset**
- Type: **Tabular**
- Target variable: `stroke` (binary)

### Quantitative Features
- age
- avg_glucose_level
- bmi

### Categorical / Binary Features
- gender
- hypertension
- heart_disease
- ever_married
- work_type
- Residence_type
- smoking_status

---

## 🔬 Statistical Analysis

For each quantitative feature, the following statistics are computed manually:

- Mean
- Median
- Mode
- Variance
- Standard Deviation
- Range

### Normality Testing
Normality is tested using the **Shapiro–Wilk test**:

- H₀: Feature follows a normal distribution
- H₁: Feature does not follow a normal distribution

---

## 🧹 Data Preprocessing Pipeline

1. Outlier Removal (Z-score method, |z| ≤ 3)
2. Train–Test Split (80% / 20%, stratified)
3. Standardization
   - Descriptive standardization (global)
   - Model standardization (training-set based)

---

## 🤖 Naive Bayes Classifier

- Gaussian Naive Bayes implemented **from scratch**
- Log-probability formulation for numerical stability
- Compared against `sklearn.naive_bayes.GaussianNB`

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## 🖥️ Graphical User Interface (PyQt5)

The GUI provides:

- One-click execution of the full pipeline
- Feature distribution visualization
- Conditional distributions P(x|y)
- Standardization visualization
- Descriptive statistics and normality test results
- Naive Bayes performance comparison

---

## 📁 Project Structure

```text
Project/
├── Dataset/
│   ├── full_data.csv
│   ├── cleaned_full_data.csv
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   ├── y_test.csv
│   ├── X_train_std.csv
│   ├── X_test_std.csv
│   └── quantitative_statistics.csv
│
├── removing_outliers.py
├── quantitative_statistics.py
├── split_data.py
├── standardization.py
├── standardization_for_model.py
├── distribution_analysis.py
├── naive_bayes.py
├── gui_pyqt5.py
└── README.md
```

---

## ▶️ How to Run the Project

### 1. Install Requirements

pip install pandas numpy scipy scikit-learn matplotlib pyqt5

### 2. Run the GUI (Recommended)

python gui_pyqt5.py

### 3. Run Individual Steps (Optional)

python removing_outliers.py
python quantitative_statistics.py
python split_data.py
python standardization.py
python standardization_for_model.py
python distribution_analysis.py
python naive_bayes.py

---

## 🧠 Design Philosophy

- Modular architecture
- Import-safe, reusable scripts
- GUI acts as an orchestrator
- No duplicated logic

---

## 📈 Results Summary

- From-scratch Naive Bayes achieves accuracy comparable to sklearn
- Overlapping feature distributions indicate realistic classification difficulty
- Standardization improves numerical stability

---

## 📝 Academic Notes

This project satisfies requirements for:

- Binary classification experiments
- Statistical feature analysis
- Naive Bayes implementation from scratch
- Visualization and interpretation
- Software engineering best practices

---

## Contributors <a name = "contributors"></a>
<table align="center">
  <tr>
    <td align="center">
      <a href="https://github.com/hamdy-cufe-eng" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/183446123?s=96&v=4" width="100px;" alt="Hamdy Ahmed"/><br />
        <sub><b>Hamdy Ahmed</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/OmegasHyper" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/180775212?v=4" width="100px;" alt="Mohamed Abdelrazek"/><br />
        <sub><b>Mohamed Abdelrazek</b></sub>
      </a>
    </td>
      <td align="center">
      <a href="https://github.com/SulaimanAlfozan" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/191874168?v=4" width="100px;" alt="Sulaiman"/><br />
        <sub><b>Sulaiman</b></sub>
      </a>
    </td>
  </tr>
  
</table>
