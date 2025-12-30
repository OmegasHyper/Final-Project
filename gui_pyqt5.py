import sys
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QTextEdit,
    QMessageBox, QTabWidget, QGroupBox, QFrame
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# ===== Pipeline Modules =====
import feature_types
import removing_outliers
import quantitative_statistics
import split_data
import standardization
import standardization_for_model
import distribution_analysis
import naive_bayes

import matplotlib as mpl

# Professional Data Science Plotting Theme
mpl.rcParams.update({
    "figure.facecolor": "#1e272e",
    "axes.facecolor": "#1e272e",
    "axes.edgecolor": "#dcdde1",
    "axes.labelcolor": "#dcdde1",
    "xtick.color": "#dcdde1",
    "ytick.color": "#dcdde1",
    "text.color": "#ffffff",
    "grid.color": "#2f3640",
    "legend.facecolor": "#2f3640",
    "legend.edgecolor": "#dcdde1"
})


class NaiveBayesDashboard(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Medical Analysis Hub – Team 16")
        self.setGeometry(50, 50, 1550, 950)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(25, 25, 25, 25)
        main_layout.setSpacing(20)

        # ================= TOP NAVIGATION / HEADER =================
        header_container = QFrame()
        header_container.setStyleSheet("background-color: #2f3640; border-radius: 10px;")
        header_layout = QHBoxLayout(header_container)

        header = QLabel("BRAIN STROKE CLASSIFICATION")
        header.setStyleSheet("font-size: 24px; font-weight: 800; color: #f5f6fa; letter-spacing: 2px;")

        team_label = QLabel("TEAM 16 | Bio-Statistics")
        team_label.setStyleSheet("font-size: 14px; color: #95a5a6;")

        header_layout.addWidget(header)
        header_layout.addStretch()
        header_layout.addWidget(team_label)

        main_layout.addWidget(header_container)

        # Load Features
        self.quant_features, self.cat_features = feature_types.get_feature_types()

        # ================= MAIN CONTENT =================
        content_layout = QHBoxLayout()
        content_layout.setSpacing(25)

        # ----- LEFT PANEL (Sidebar) -----
        left_panel = QGroupBox("DATA PARAMETERS")
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(15, 25, 15, 15)

        # Quantitative List
        left_layout.addWidget(QLabel("<b style='color: #48dbfb;'>NUMERIC FEATURES</b>"))
        q_label = QLabel("\n".join([f"  {f}" for f in self.quant_features]))
        q_label.setStyleSheet("color: #dcdde1; font-size: 14px; padding-bottom: 10px;")
        left_layout.addWidget(q_label)

        # Categorical List
        left_layout.addWidget(QLabel("<b style='color: #ff9f43;'>CATEGORICAL FEATURES</b>"))
        c_label = QLabel("\n".join([f"  {f}" for f in self.cat_features]))
        c_label.setStyleSheet("color: #dcdde1; font-size: 14px; padding-bottom: 20px;")
        left_layout.addWidget(c_label)

        # Selection
        left_layout.addWidget(QLabel("PRIMARY ANALYSIS FEATURE:"))
        self.feature_box = QComboBox()
        self.feature_box.addItems(self.quant_features)
        left_layout.addWidget(self.feature_box)

        left_layout.addStretch()

        self.run_btn = QPushButton("EXECUTE PIPELINE")
        self.run_btn.clicked.connect(self.run_pipeline)
        left_layout.addWidget(self.run_btn)

        left_panel.setLayout(left_layout)
        content_layout.addWidget(left_panel, 1)

        # ----- RIGHT PANEL (Tabs) -----
        self.tabs = QTabWidget()

        # 1. Visualization
        self.plot_tab = QWidget()
        plot_tab_layout = QVBoxLayout()
        self.fig = plt.figure(figsize=(10, 7))
        self.canvas = FigureCanvas(self.fig)
        plot_tab_layout.addWidget(self.canvas)
        self.plot_tab.setLayout(plot_tab_layout)

        # 2. Statistics
        self.stats_tab = QTextEdit(readOnly=True)

        # 3. Normality
        self.normality_tab = QTextEdit(readOnly=True)

        # 4. Results
        self.model_tab = QTextEdit(readOnly=True)

        self.tabs.addTab(self.plot_tab, "VISUALIZATION")
        self.tabs.addTab(self.stats_tab, "DESCRIPTIVE STATS")
        self.tabs.addTab(self.normality_tab, "NORMALITY ANALYSIS")
        self.tabs.addTab(self.model_tab, "MODEL METRICS")

        content_layout.addWidget(self.tabs, 4)
        main_layout.addLayout(content_layout)

        self.setLayout(main_layout)

    # ================= LOGIC =================

    def run_pipeline(self):
        try:
            removing_outliers.run();
            quantitative_statistics.run();
            split_data.run()
            standardization.run();
            standardization_for_model.run()
            distribution_analysis.run();
            naive_bayes.run()

            self.update_statistics();
            self.update_plots()
            self.update_normality();
            self.update_model_results()
            QMessageBox.information(self, "Status", "Optimization Pipeline Completed.")
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"System Failure: {str(e)}")

    def update_statistics(self):
        feature = self.feature_box.currentText()
        stats_df = pd.read_csv("Dataset/quantitative_statistics.csv")
        row = stats_df[stats_df.Feature == feature].iloc[0]

        html = f"""
        <div style="font-family: 'Segoe UI'; padding: 30px; color: #ecf0f1;">
            <h1 style="color: #48dbfb; border-bottom: 2px solid #2f3640;">Feature Profile: {feature}</h1>
            <table width="100%" style="font-size: 18px; margin-top: 20px;" cellpadding="10">
                <tr style="background-color: #2f3640;"><td><b>Metric</b></td><td><b>Value</b></td></tr>
                <tr><td>Arithmatic Mean</td><td>{row['Mean']:.4f}</td></tr>
                <tr><td>Median Value</td><td>{row['Median']:.4f}</td></tr>
                <tr><td>Mode</td><td>{row['Mode']:.4f}</td></tr>
                <tr><td>Variance</td><td>{row['Variance']:.4f}</td></tr>
                <tr><td>Standard Deviation</td><td>{row['Std_Deviation']:.4f}</td></tr>
                <tr><td>Range</td><td>{row['Range']:.4f}</td></tr>
            </table>
        </div>
        """
        self.stats_tab.setHtml(html)

    def update_plots(self):
        feature = self.feature_box.currentText()
        train_data = pd.read_csv("Dataset/X_train_std.csv").join(pd.read_csv("Dataset/y_train.csv"))

        self.fig.clear()
        colors = ['#54a0ff', '#1dd1a1', '#ee5253', '#feca57']

        ax1 = self.fig.add_subplot(221);
        ax1.hist(train_data[feature], bins=40, color=colors[0], edgecolor='#2f3542');
        ax1.set_title("Full Distribution")
        ax2 = self.fig.add_subplot(222);
        ax2.hist(train_data[train_data.stroke == 0][feature], bins=40, density=True, color=colors[1], alpha=0.7);
        ax2.set_title("P(x | No Stroke)")
        ax3 = self.fig.add_subplot(223);
        ax3.hist(train_data[train_data.stroke == 1][feature], bins=40, density=True, color=colors[2], alpha=0.7);
        ax3.set_title("P(x | Stroke)")
        ax4 = self.fig.add_subplot(224);
        stats.probplot(train_data[feature], dist="norm", plot=ax4);
        ax4.set_title("Quantile-Quantile Plot")

        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()

    def update_normality(self):
        feat = self.feature_box.currentText()
        data = pd.read_csv("Dataset/X_train.csv")[feat].dropna()
        stat, p = stats.shapiro(data.sample(min(3000, len(data))))

        html = f"""
        <div style="font-family: 'Segoe UI'; padding: 30px; color: #ecf0f1;">
            <h1 style="color: #48dbfb;">Inferential Statistics: {feat}</h1>
            <p style="font-size: 16px; color: #bdc3c7;"><b>Hypothesis Test:</b> Shapiro-Wilk Normality Test</p>
            <div style="background-color: #2f3640; padding: 20px; border-radius: 10px;">
                <p><b>P-Value:</b> <span style="color: #ff6b6b; font-size: 22px;">{p:.4e}</span></p>
                <p><b>Status:</b> {'Reject Null Hypothesis (Non-Normal)' if p < 0.05 else 'Fail to Reject Null (Normal)'}</p>
            </div>
            <p style="margin-top: 20px;"><i>Note: A p-value less than 0.05 indicates the data is significantly different from a normal distribution.</i></p>
        </div>
        """
        self.normality_tab.setHtml(html)

    def update_model_results(self):
        # Implementation of metrics logic...
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
        p, r, f1 = nb.precision_recall_f1(y_test, y_pred)

        sk = GaussianNB()
        sk.fit(X_train, y_train)
        y_pred_sk = sk.predict(X_test)

        self.model_tab.setHtml(
            f"""
            <h1 style='color: #48dbfb; padding: 30px;'>Naive Bayes Performance.</h1>

            <table width="100%" style="font-size: 18px; margin-top: 20px;" cellpadding="10">
                <tr style="background-color: #2f3640;">
                    <td><b>POC.</b></td>
                    <td><b>GaussianNB From Scratch</b></td>
                    <td><b>Sklearn GaussianNB</b></td>
                </tr>
                <tr>
                    <td>Accuracy</td>
                    <td>{accuracy_score(y_test, y_pred):.4f}</td>
                    <td>{accuracy_score(y_test, y_pred_sk):.4f}</td>
                </tr>
                <tr>
                    <td>Precision</td>
                    <td>{precision_score(y_test, y_pred, average='macro'):.4f}</td>
                    <td>{precision_score(y_test, y_pred, average='macro'):.4f}</td>
                </tr>
                <tr>
                    <td>Recall</td>
                    <td>{r:.4f}</td>
                    <td>{recall_score(y_test, y_pred, average='macro'):.4f}</td>
                </tr>
                <tr>
                    <td>F1-score</td>
                    <td>{f1:.4f}</td>
                    <td>{f1_score(y_test, y_pred, average='macro'):.4f}</td>
                </tr>
            </table>
            """)


# ================= PROFESSIONAL STYLESHEET =================
stylesheet = """
QWidget { background-color: #1e272e; color: #f5f6fa; font-family: 'Segoe UI', Arial; }
QGroupBox { border: 1px solid #4b7bec; border-radius: 8px; margin-top: 15px; font-weight: bold; font-size: 13px; color: #4b7bec; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
QTabWidget::pane { border: 1px solid #2f3640; background: #1e272e; }
QTabBar::tab { background: #2f3640; padding: 15px 35px; border-top-left-radius: 4px; border-top-right-radius: 4px; font-size: 13px; margin-right: 2px; }
QTabBar::tab:selected { background: #4b7bec; color: white; border-bottom: 2px solid #48dbfb; }
QComboBox { background: #2f3640; border: 1px solid #4b7bec; padding: 8px; border-radius: 4px; min-width: 150px; }
QPushButton { background-color: #27ae60; border-radius: 5px; padding: 12px; font-weight: bold; font-size: 14px; color: white; }
QPushButton:hover { background-color: #2ecc71; }
QTextEdit { background-color: #1e272e; border: none; }
"""

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(stylesheet)
    window = NaiveBayesDashboard()
    window.show()
    sys.exit(app.exec_())