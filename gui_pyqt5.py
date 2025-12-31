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

        # 1. Window Configuration
        self.setWindowTitle("Medical Analysis Hub – Team 16")
        self.setGeometry(50, 50, 1550, 950)

        # 2. Rounded & Frameless Setup
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)  # Essential for rounded transparency

        # 3. Main Outer Layout
        master_layout = QVBoxLayout(self)
        master_layout.setContentsMargins(0, 0, 0, 0)

        # 4. The Rounded Container (This becomes your "window" visually)
        self.main_container = QFrame()
        self.main_container.setObjectName("MainContainer")
        container_layout = QVBoxLayout(self.main_container)
        container_layout.setContentsMargins(25, 25, 25, 25)
        container_layout.setSpacing(20)

        # ================= TOP NAVIGATION / HEADER =================
        header_container = QFrame()
        header_container.setStyleSheet("background-color: #2f3640; border-radius: 10px;")
        header_layout = QHBoxLayout(header_container)

        header = QLabel("BRAIN STROKE CLASSIFICATION")
        header.setStyleSheet("font-size: 24px; font-weight: 800; color: #f5f6fa; letter-spacing: 2px;")

        team_label = QLabel("TEAM 16 | Bio-Statistics")
        team_label.setStyleSheet("font-size: 14px; color: #95a5a6; margin-right: 20px;")

        self.exit_btn = QPushButton("✕")
        self.exit_btn.setFixedSize(35, 35)
        self.exit_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c; 
                border-radius: 17px; 
                font-weight: bold;
                font-size: 16px;
                border: none;
                color: white;
            }
            QPushButton:hover { background-color: #c0392b; }
        """)
        self.exit_btn.clicked.connect(self.close)

        header_layout.addWidget(header)
        header_layout.addStretch()
        header_layout.addWidget(team_label)
        header_layout.addWidget(self.exit_btn)

        container_layout.addWidget(header_container)

        # Load Features
        self.quant_features, self.cat_features = feature_types.get_feature_types()

        # ================= MAIN CONTENT =================
        content_layout = QHBoxLayout()
        content_layout.setSpacing(25)

        # ----- LEFT PANEL (Sidebar) -----
        left_panel = QGroupBox("DATA PARAMETERS")
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(15, 25, 15, 15)

        left_layout.addWidget(QLabel("<b style='color: #ff9f43;'>QUANTITATIVE FEATURES</b>"))
        q_label = QLabel("\n".join([f"  {f}" for f in self.quant_features]))
        q_label.setStyleSheet("color: #dcdde1; font-size: 14px; padding-bottom: 10px;")
        left_layout.addWidget(q_label)

        left_layout.addWidget(QLabel("<b style='color: #ff9f43;'>CATEGORICAL FEATURES</b>"))
        c_label = QLabel("\n".join([f"  {f}" for f in self.cat_features]))
        c_label.setStyleSheet("color: #dcdde1; font-size: 14px; padding-bottom: 20px;")
        left_layout.addWidget(c_label)

        left_layout.addWidget(QLabel("<b style='color: #ff9f43;'>ANALYZE FEATURES</b>"))
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
        self.plot_tab = QWidget()
        plot_tab_layout = QVBoxLayout()
        self.fig = plt.figure(figsize=(10, 7))
        self.canvas = FigureCanvas(self.fig)
        plot_tab_layout.addWidget(self.canvas)
        self.plot_tab.setLayout(plot_tab_layout)

        self.stats_tab = QTextEdit(readOnly=True)
        self.normality_tab = QTextEdit(readOnly=True)
        self.model_tab = QTextEdit(readOnly=True)

        self.tabs.addTab(self.plot_tab, "VISUALIZATION")
        self.tabs.addTab(self.stats_tab, "DESCRIPTIVE STATS")
        self.tabs.addTab(self.normality_tab, "NORMALITY ANALYSIS")
        self.tabs.addTab(self.model_tab, "MODEL METRICS")

        content_layout.addWidget(self.tabs, 4)
        container_layout.addLayout(content_layout)

        # Add the rounded container to the master window layout
        master_layout.addWidget(self.main_container)

    # ================= MOUSE EVENTS FOR DRAGGING =================
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.old_pos = event.globalPos()

    def mouseMoveEvent(self, event):
        if hasattr(self, 'old_pos'):
            delta = event.globalPos() - self.old_pos
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.old_pos = event.globalPos()

    def mouseReleaseEvent(self, event):
        if hasattr(self, 'old_pos'):
            del self.old_pos

    # ================= LOGIC FUNCTIONS =================
    def run_pipeline(self):
        """
        Execute the complete pipeline in the correct order per requirements:
        1. Remove outliers
        2. Calculate descriptive statistics
        3. Standardize features (descriptive, using full data)
        4. Split data (80% train, 20% test)
        5. Training Data Feature Analysis (histograms, distribution types, normality tests, conditional distributions)
        6. Standardize for model (using training statistics only)
        7. Apply Naive Bayes classifier
        """
        try:
            # Step 1: Remove outliers
            removing_outliers.run()
            
            # Step 2: Calculate descriptive statistics
            quantitative_statistics.run()
            
            # Step 3: Standardize features (for descriptive analysis, using full data)
            standardization.run()
            
            # Step 4: Split data into training (80%) and testing (20%)
            split_data.run()
            
            # Step 5: Training Data Feature Analysis
            # - Plot histograms/distributions
            # - Comment on distribution types
            # - Statistical test for normality
            # - Plot conditional distributions P(x|y)
            distribution_analysis.run()
            
            # Step 6: Standardize for model (using training statistics only)
            standardization_for_model.run()
            
            # Step 7: Apply Naive Bayes classifier
            naive_bayes.run()

            # Update GUI displays
            self.update_statistics()
            self.update_plots()
            self.update_normality()
            self.update_model_results()
            
            QMessageBox.information(self, "Success", "More Conditional Plots are saved to the Dataset/plots")
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"System Failure: {str(e)}")

    def update_statistics(self):
        feature = self.feature_box.currentText()
        stats_df = pd.read_csv("Dataset/quantitative_statistics.csv")
        row = stats_df[stats_df.Feature == feature].iloc[0]

        html = f"""
        <div style="font-family: 'Segoe UI'; padding: 30px; color: #ecf0f1;">
            <h1 style="color: #ff9f43; border-bottom: 2px solid #2f3640;">Feature Profile: {feature.title()}</h1>
            <table width="100%" style="font-size: 18px; margin-top: 20px;" cellpadding="10">
                <tr style="background-color: #2f3640;"><td><b>Metric</b></td><td><b>Value</b></td></tr>
                <tr><td>Arithmetic Mean</td><td>{row['Mean']:.4f}</td></tr>
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
        """
        Display distribution plots for the selected feature.
        Uses standardized training data (what the model uses).
        """
        feature = self.feature_box.currentText()
        # Use standardized training data for visualization (what the model sees)
        train_data = pd.read_csv("Dataset/X_train_std.csv").join(pd.read_csv("Dataset/y_train.csv"))
        self.fig.clear()
        colors = ['#54a0ff', '#1dd1a1', '#ee5253', '#feca57']

        ax1 = self.fig.add_subplot(221);
        ax1.hist(train_data[feature], bins=40, color=colors[0], edgecolor='#2f3542');
        ax1.set_title(f"Full Distribution: {feature.title()} (Standardized)")
        ax1.set_xlabel(f"{feature} (standardized)")
        ax1.set_ylabel('Frequency')

        ax2 = self.fig.add_subplot(222);
        ax2.hist(train_data[train_data.stroke == 0][feature], bins=40, density=True, color=colors[1], alpha=0.7);
        ax2.set_title("P(x | No Stroke)")
        ax2.set_xlabel(f"{feature} (standardized)")
        ax2.set_ylabel('Density')

        ax3 = self.fig.add_subplot(223);
        ax3.hist(train_data[train_data.stroke == 1][feature], bins=40, density=True, color=colors[2], alpha=0.7);
        ax3.set_title("P(x | Stroke)")
        ax3.set_xlabel(f"{feature} (standardized)")
        ax3.set_ylabel('Density')

        ax4 = self.fig.add_subplot(224);
        stats.probplot(train_data[feature], dist="norm", plot=ax4);
        ax4.set_title("Quantile-Quantile Plot (Standardized)")
        ax4.set_xlabel('Theoretical Quantiles')
        ax4.set_ylabel('Sample Quantiles')

        self.fig.tight_layout(pad=3.0);
        self.canvas.draw()

    def update_normality(self):
        """
        Display distribution analysis results from distribution_analysis.py in the normality test tab.
        Shows distribution type, skewness, kurtosis, and Shapiro-Wilk test results.
        """
        feat = self.feature_box.currentText()
        
        try:
            # Try to load results from distribution analysis CSV
            results_df = pd.read_csv("Dataset/distribution_analysis_results.csv")
            row = results_df[results_df['Feature'] == feat].iloc[0]
            
            # Extract values
            dist_type = row['Distribution_Type']
            skewness = row['Skewness']
            kurtosis = row['Kurtosis']
            shapiro_stat = row['Shapiro_Statistic']
            shapiro_p = row['Shapiro_p_value']
            is_normal = row['Is_Normal']
            no_stroke_count = int(row['No_Stroke_Count'])
            stroke_count = int(row['Stroke_Count'])
            
            # Determine status color and verdict
            alpha = 0.05
            if is_normal:
                status_color = "#27ae60"
                verdict_text = "FAIL TO REJECT H₀"
                verdict_detail = f"(p-value = {shapiro_p:.6e} > α = {alpha})"
                conclusion = "The feature appears to follow a normal distribution."
            else:
                status_color = "#e74c3c"
                verdict_text = "REJECT H₀"
                verdict_detail = f"(p-value = {shapiro_p:.6e} ≤ α = {alpha})"
                conclusion = "The feature does NOT follow a normal distribution."
            
            html = f"""
            <div style="font-family: 'Segoe UI'; padding: 25px; color: #ecf0f1;">
                <h1 style="color: #ff9f43; border-bottom: 2px solid #2f3640; padding-bottom: 10px;">
                    DISTRIBUTION ANALYSIS: {feat.upper()}
                </h1>

                <!-- Distribution Type Section -->
                <div style="margin-top: 25px; background-color: #2f3640; padding: 20px; border-radius: 10px; border-left: 5px solid #ff9f43;">
                    <h2 style="color: #ff9f43; margin-top: 0;">1. DISTRIBUTION TYPE</h2>
                    <p style="font-size: 18px; font-weight: bold; margin: 10px 0;">{dist_type}</p>
                    <div style="margin-top: 15px; display: flex; gap: 20px;">
                        <div>
                            <span style="color: #95a5a6;">Skewness:</span>
                            <span style="font-weight: bold; margin-left: 10px;">{skewness:.4f}</span>
                        </div>
                        <div>
                            <span style="color: #95a5a6;">Kurtosis:</span>
                            <span style="font-weight: bold; margin-left: 10px;">{kurtosis:.4f}</span>
                        </div>
                    </div>
                </div>

                <!-- Normality Test Section -->
                <div style="margin-top: 20px; background-color: #2f3640; padding: 20px; border-radius: 10px; border-left: 5px solid #54a0ff;">
                    <h2 style="color: #54a0ff; margin-top: 0;">2. NORMALITY TEST: Shapiro-Wilk Test</h2>
                    
                    <div style="margin-top: 15px; background-color: #1e272e; padding: 15px; border-radius: 8px;">
                        <h3 style="color: #ff9f43; margin-top: 0; font-size: 16px;">Hypotheses:</h3>
                        <p style="margin: 8px 0; line-height: 1.8;">
                            <b>H₀ (Null Hypothesis):</b> The feature '{feat}' follows a normal distribution<br>
                            <b>H₁ (Alternative Hypothesis):</b> The feature '{feat}' does NOT follow a normal distribution
                        </p>
                    </div>

                    <div style="margin-top: 20px; display: flex; gap: 15px;">
                        <div style="flex: 1; padding: 15px; background-color: #1e272e; border-radius: 10px; text-align: center;">
                            <p style="margin: 0; font-size: 11px; color: #95a5a6;">TEST STATISTIC (W)</p>
                            <p style="margin: 5px 0 0 0; font-size: 22px; font-weight: bold;">{shapiro_stat:.6f}</p>
                        </div>
                        <div style="flex: 1; padding: 15px; background-color: #1e272e; border-radius: 10px; text-align: center;">
                            <p style="margin: 0; font-size: 11px; color: #95a5a6;">p-VALUE</p>
                            <p style="margin: 5px 0 0 0; font-size: 20px; font-weight: bold;">{shapiro_p:.6e}</p>
                        </div>
                        <div style="flex: 1; padding: 15px; background-color: #1e272e; border-radius: 10px; text-align: center;">
                            <p style="margin: 0; font-size: 11px; color: #95a5a6;">SIGNIFICANCE LEVEL (α)</p>
                            <p style="margin: 5px 0 0 0; font-size: 22px; font-weight: bold;">{alpha}</p>
                        </div>
                    </div>

                    <div style="margin-top: 20px; padding: 15px; background-color: #1e272e; border-radius: 10px; border: 2px solid {status_color};">
                        <p style="margin: 0; font-size: 14px; color: #95a5a6; text-align: center;">VERDICT</p>
                        <p style="margin: 5px 0 0 0; font-size: 20px; font-weight: bold; color: {status_color}; text-align: center;">
                            {verdict_text}
                        </p>
                        <p style="margin: 5px 0 0 0; font-size: 12px; color: #95a5a6; text-align: center;">
                            {verdict_detail}
                        </p>
                    </div>

                    <div style="margin-top: 15px; padding: 12px; background-color: #161b22; border-radius: 8px;">
                        <p style="margin: 0; line-height: 1.6; font-size: 14px;">
                            <b style="color: #ff9f43;">Conclusion:</b> {conclusion}
                        </p>
                    </div>
                </div>

                <!-- Conditional Distribution Info -->
                <div style="margin-top: 20px; background-color: #2f3640; padding: 20px; border-radius: 10px; border-left: 5px solid #1dd1a1;">
                    <h2 style="color: #1dd1a1; margin-top: 0;">3. CONDITIONAL DISTRIBUTIONS</h2>
                    <div style="margin-top: 15px; display: flex; gap: 30px;">
                        <div>
                            <span style="color: #95a5a6;">P(x | No Stroke):</span>
                            <span style="font-weight: bold; margin-left: 10px; color: #1dd1a1;">{no_stroke_count:,} samples</span>
                        </div>
                        <div>
                            <span style="color: #95a5a6;">P(x | Stroke):</span>
                            <span style="font-weight: bold; margin-left: 10px; color: #ee5253;">{stroke_count:,} samples</span>
                        </div>
                        <div>
                        Conditional distributions show how the feature varies within each target class. 
                        Visual plots are available in the Visualization tab and saved to <code>Dataset/plots/{feat}_distribution_analysis.png</code>
                        </div>
                    </div>
                  
                </div>

             
            </div>
            """
            
        except (FileNotFoundError, IndexError, KeyError) as e:
            # Fallback if CSV doesn't exist or feature not found
            html = f"""
            <div style="font-family: 'Segoe UI'; padding: 25px; color: #ecf0f1;">
                <h1 style="color: #ff9f43; border-bottom: 2px solid #2f3640; padding-bottom: 10px;">
                    NORMALITY ANALYSIS: {feat.upper()}
                </h1>
                <div style="margin-top: 25px; padding: 20px; background-color: #e74c3c; border-radius: 10px;">
                    <p style="margin: 0; font-size: 16px;">
                        <b>Distribution analysis not yet completed.</b><br>
                        Please run the pipeline first to generate distribution analysis results.
                    </p>
                    <p style="margin-top: 15px; font-size: 14px; color: #f5f6fa;">
                        Click "EXECUTE PIPELINE" to perform the full analysis.
                    </p>
                </div>
            </div>
            """
        
        self.normality_tab.setHtml(html)
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
        acc, p, r, f1 = nb.precision_recall_f1(y_test, y_pred)

        sk = GaussianNB()
        sk.fit(X_train, y_train)
        y_pred_sk = sk.predict(X_test)

        self.model_tab.setHtml(
            f"""
            <h1 style='color: #ff9f43; padding: 30px;'>Naive Bayes Performance.</h1>

            <table width="100%" style="font-size: 18px; margin-top: 20px;" cellpadding="10">
                <tr style="background-color: #2f3640;">
                    <td><b>POC.</b></td>
                    <td><b>GaussianNB From Scratch</b></td>
                    <td><b>Sklearn GaussianNB</b></td>
                </tr>
                <tr>
                    <td>Accuracy</td>
                    <td>{acc:.4f}</td>
                    <td>{accuracy_score(y_test, y_pred_sk):.4f}</td>
                </tr>
                <tr>
                    <td>Precision</td>
                    <td>{p:.4f}</td>
                    <td>{precision_score(y_test, y_pred_sk, average='macro'):.4f}</td>
                </tr>
                <tr>
                    <td>Recall</td>
                    <td>{r:.4f}</td>
                    <td>{recall_score(y_test, y_pred_sk, average='macro'):.4f}</td>
                </tr>
                <tr>
                    <td>F1-score</td>
                    <td>{f1:.4f}</td>
                    <td>{f1_score(y_test, y_pred_sk, average='macro'):.4f}</td>
                </tr>
            </table>
            """)


# ================= ENHANCED STYLESHEET =================
stylesheet = """
/* The Main Rounded Container */
QFrame#MainContainer { 
    background-color: #1e272e; 
    border: 1px solid #2f3640;
    border-radius: 25px; 
}

QWidget { color: #f5f6fa; font-family: 'Segoe UI', Arial; }
QGroupBox { border: 1px solid #2f3640; border-radius: 8px; margin-top: 15px; font-weight: bold; font-size: 13px; color: #fff; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
QTabWidget::pane { border: 1px solid #2f3640; background: #1e272e; border-radius: 5px; }
QTabBar::tab { background: #2f3640; padding: 15px 35px; border-top-left-radius: 4px; border-top-right-radius: 4px; font-size: 13px; margin-right: 2px; }
QTabBar::tab:selected { background: #2f3640; color: white; border-bottom: 2px solid #ff9f43; }
QComboBox { background: #2f3640; border: 1px solid #2f3640; padding: 8px; border-radius: 4px; min-width: 150px; color: white; }
QComboBox QAbstractItemView {
    background-color: #2f3640;
    color: white;
    selection-background-color: #ff9f43; /* Highlight color when hovering */
    selection-color: #1e272e;           /* Text color when highlighted */
    border: 1px solid #ff9f43;
    outline: none;
}
QPushButton { background-color: #2f3640; border-radius: 5px; padding: 12px; font-weight: bold; font-size: 14px; color: white; border: 1px solid #2f3640; }
QPushButton:hover { background-color: #ff9f43; color: #1e272e; }
QTextEdit { background-color: #1e272e; border: none; }
QMessageBox { background-color: #1e272e; border: 1px solid #2f3640; border-radius: 10px; padding: 20px; }
QMessageBox::icon { color: #ff9f43; }
QMessageBox::title { color: #ff9f43; }
QMessageBox::text { color: #f5f6fa; }
QMessageBox::button { background-color: #2f3640; border: 1px solid #2f3640; border-radius: 5px; padding: 10px; font-weight: bold; font-size: 14px; color: white; }
QMessageBox::button:hover { background-color: #ff9f43; color: #1e272e; }
QMessageBox::button:pressed { background-color: #e74c3c; color: #1e272e; }
QMessageBox::button:disabled { background-color: #2f3640; color: #95a5a6; }
"""

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(stylesheet)
    window = NaiveBayesDashboard()
    window.show()
    sys.exit(app.exec_())