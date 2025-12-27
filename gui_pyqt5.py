import sys
import pandas as pd
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QTextEdit,
    QMessageBox, QTabWidget, QGroupBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# ===== Pipeline Modules =====
import removing_outliers
import quantitative_statistics
import split_data
import standardization
import standardization_for_model
import distribution_analysis
import naive_bayes

import matplotlib as mpl

mpl.rcParams.update({
    "figure.facecolor": "#1e1e1e",   # rgb(30,30,30)
    "axes.facecolor": "#1e1e1e",
    "axes.edgecolor": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "text.color": "white",
    "grid.color": "gray",
    "legend.facecolor": "#1e1e1e",
    "legend.edgecolor": "white"
})
class NaiveBayesDashboard(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Naive Bayes Stroke Classification – Team 16")
        self.setGeometry(80, 80, 1500, 900)

        main_layout = QVBoxLayout()

        # ================= HEADER =================
        header = QLabel("Naive Bayes Stroke Classification - Team 16")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("""
            font-size: 22px;
            font-weight: bold;
            padding: 10px;
        """)
        main_layout.addWidget(header)

        # ================= TOP CONTENT =================
        top_layout = QHBoxLayout()

        # ----- LEFT PANEL (Controls) -----
        left_panel = QGroupBox("Controls")
        left_layout = QVBoxLayout()

        self.feature_box = QComboBox()
        self.feature_box.addItems(['age', 'avg_glucose_level', 'bmi'])

        self.run_btn = QPushButton("Run Full Pipeline")
        self.run_btn.setStyleSheet("padding: 8px; font-weight: bold;")
        self.run_btn.clicked.connect(self.run_pipeline)

        left_layout.addWidget(QLabel("Select Feature"))
        left_layout.addWidget(self.feature_box)
        left_layout.addStretch()
        left_layout.addWidget(self.run_btn)

        left_panel.setLayout(left_layout)

        # ----- RIGHT PANEL (Plots) -----
        right_panel = QGroupBox("Data Visualization")
        right_layout = QVBoxLayout()

        self.fig = plt.figure(figsize=(10, 7))
        self.canvas = FigureCanvas(self.fig)

        right_layout.addWidget(self.canvas)
        right_panel.setLayout(right_layout)

        top_layout.addWidget(left_panel, 1)
        top_layout.addWidget(right_panel, 4)

        main_layout.addLayout(top_layout)

        # ================= TABS =================
        self.tabs = QTabWidget()

        self.stats_tab = QTextEdit()
        self.stats_tab.setReadOnly(True)

        self.normality_tab = QTextEdit()
        self.normality_tab.setReadOnly(True)

        self.model_tab = QTextEdit()
        self.model_tab.setReadOnly(True)

        self.tabs.addTab(self.stats_tab, "Descriptive Statistics")
        self.tabs.addTab(self.normality_tab, "Normality Tests")
        self.tabs.addTab(self.model_tab, "Naive Bayes Results")

        main_layout.addWidget(self.tabs)

        self.setLayout(main_layout)

    # =====================================================
    # PIPELINE EXECUTION
    # =====================================================
    def run_pipeline(self):
        try:
            removing_outliers.run()
            quantitative_statistics.run()
            split_data.run()
            standardization.run()
            standardization_for_model.run()
            distribution_analysis.run()
            naive_bayes.run()

            self.update_statistics()
            self.update_plots()
            self.update_model_results()

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    # =====================================================
    # STATISTICS TAB
    # =====================================================
    def update_statistics(self):
        feature = self.feature_box.currentText()
        stats_df = pd.read_csv("Dataset/quantitative_statistics.csv")
        row = stats_df[stats_df.Feature == feature].iloc[0]

        self.stats_tab.setText(
            f"Feature: {feature}\n\n"
            f"Mean: {row['Mean']:.4f}\n"
            f"Median: {row['Median']:.4f}\n"
            f"Mode: {row['Mode']:.4f}\n"
            f"Variance: {row['Variance']:.4f}\n"
            f"Standard Deviation: {row['Std_Deviation']:.4f}\n"
            f"Range: {row['Range']:.4f}"
        )

    # =====================================================
    # PLOTS
    # =====================================================
    def update_plots(self):
        feature = self.feature_box.currentText()

        df = pd.read_csv("Dataset/cleaned_full_data.csv")
        X_train_std = pd.read_csv("Dataset/X_train_std.csv")

        self.fig.clear()

        ax1 = self.fig.add_subplot(221)
        ax2 = self.fig.add_subplot(222)
        ax3 = self.fig.add_subplot(223)
        ax4 = self.fig.add_subplot(224)

        ax1.hist(df[feature], bins=30)
        ax1.set_title("Overall Distribution")

        ax2.hist(df[df.stroke == 0][feature], bins=30, density=True)
        ax2.set_title("P(x | No Stroke)")

        ax3.hist(df[df.stroke == 1][feature], bins=30, density=True)
        ax3.set_title("P(x | Stroke)")

        ax4.hist(X_train_std[feature], bins=30)
        ax4.set_title("After Standardization")

        self.fig.subplots_adjust(wspace=0.3, hspace=0.4)
        self.canvas.draw()

    # =====================================================
    # MODEL TAB
    # =====================================================
    def update_model_results(self):
        X_train = pd.read_csv("Dataset/X_train_std.csv").values
        X_test = pd.read_csv("Dataset/X_test_std.csv").values
        y_train = pd.read_csv("Dataset/y_train.csv").values.ravel()
        y_test = pd.read_csv("Dataset/y_test.csv").values.ravel()

        from naive_bayes import GaussianNaiveBayes
        from sklearn.naive_bayes import GaussianNB
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        nb = GaussianNaiveBayes()
        nb.fit(X_train, y_train)
        y_pred = nb.predict(X_test)

        sk = GaussianNB()
        sk.fit(X_train, y_train)
        y_pred_sk = sk.predict(X_test)

        self.model_tab.setText(
            "Naive Bayes Performance\n\n"
            f"From Scratch:\n"
            f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n"
            f"Precision: {precision_score(y_test, y_pred):.4f}\n"
            f"Recall: {recall_score(y_test, y_pred):.4f}\n"
            f"F1-score: {f1_score(y_test, y_pred):.4f}\n\n"
            f"Sklearn GaussianNB Accuracy: "
            f"{accuracy_score(y_test, y_pred_sk):.4f}"
        )

stylesheet = """
QWidget {
    background-color: rgb(30,30,30);
    color: white;
}

QLabel {
    color: white;
    font-size: 14px;
}

QPushButton {
    color: white;
    background-color: rgb(45,45,45);
    border: 1px solid rgb(9, 132, 227);
    padding: 6px;
}

QPushButton:hover {
    background-color: rgb(60,60,60);
}

QGroupBox {
    border: 1px solid rgb(9, 132, 227);
    margin-top: 10px;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: rgb(9, 132, 227);
}

QTabWidget::pane {
    border: 1px solid rgb(9, 132, 227);
}

QTextEdit {
    border: 1px solid rgb(9, 132, 227);
    background-color: rgb(25,25,25);
}

QScrollBar:vertical {
    background: rgb(30,30,30);
    width: 12px;
}

QScrollBar::handle:vertical {
    background: rgb(9, 132, 227);
    min-height: 20px;
}
"""


# =====================================================
# RUN APP
# =====================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet(stylesheet)
    window = NaiveBayesDashboard()
    window.show()
    sys.exit(app.exec_())
