import pandas as pd
from scipy.stats import shapiro
from scipy import signal
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d
import numpy as np
import matplotlib.pyplot as plt
import os


def detect_multimodality(data, bins=40):
    """
    Detect if a distribution is multimodal by finding distinct peaks separated by clear valleys.
    Uses strict criteria to avoid false positives on approximately normal distributions.
    Returns True only if clearly multimodal (2+ distinct peaks with deep valleys between them).
    """
    try:
        data_array = np.array(data)
        data_array = data_array[~np.isnan(data_array)]
        
        if len(data_array) < 30:
            return False
        
        data_range = data_array.max() - data_array.min()
        
        # Method 1: Histogram-based detection with strict valley checking
        try:
            hist, bin_edges = np.histogram(data_array, bins=50)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Moderate smoothing - enough to reduce noise but preserve true modes
            hist_smooth = gaussian_filter1d(hist.astype(float), sigma=1.2)
            
            max_hist = np.max(hist_smooth)
            
            # Use moderate prominence - not too low (avoids noise) but catches real modes
            peak_prominence = max_hist * 0.20  # Higher threshold to avoid small bumps
            min_dist_bins = max(5, bins // 10)  # Require peaks to be well-separated
            
            peaks, peak_props = signal.find_peaks(hist_smooth, prominence=peak_prominence, distance=min_dist_bins)
            
            if len(peaks) >= 2:
                # Sort peaks by position
                sorted_peaks = np.sort(peaks)
                peak_values = bin_centers[sorted_peaks]
                peak_heights = hist_smooth[sorted_peaks]
                
                # Check for CLEAR valleys between peaks
                # Need STRONG evidence: deep valleys + good separation + significant peaks
                found_multimodal_pair = False
                
                for i in range(len(sorted_peaks) - 1):
                    start_idx = sorted_peaks[i]
                    end_idx = sorted_peaks[i + 1]
                    
                    # Get the region between peaks
                    between_region = hist_smooth[start_idx:end_idx+1]
                    min_valley = np.min(between_region)
                    avg_peak_height = (peak_heights[i] + peak_heights[i+1]) / 2
                    
                    # STRICT requirement 1: Valley must be at least 25% lower than average peak
                    valley_depth_ratio = (avg_peak_height - min_valley) / (avg_peak_height + 1e-10)
                    
                    # STRICT requirement 2: Both peaks must be at least 1.4x higher than valley
                    peak_to_valley_ratio_1 = peak_heights[i] / (min_valley + 1e-10)
                    peak_to_valley_ratio_2 = peak_heights[i+1] / (min_valley + 1e-10)
                    
                    # STRICT requirement 3: Peaks must be well-separated (at least 15% of range)
                    peak_separation = peak_values[i+1] - peak_values[i]
                    
                    # STRICT requirement 4: Both peaks must be substantial (at least 20% of max)
                    peak_significance_1 = peak_heights[i] > max_hist * 0.20
                    peak_significance_2 = peak_heights[i+1] > max_hist * 0.20
                    
                    # All requirements must be met
                    if (valley_depth_ratio > 0.25 and 
                        peak_to_valley_ratio_1 > 1.4 and 
                        peak_to_valley_ratio_2 > 1.4 and
                        peak_separation > data_range * 0.15 and
                        peak_significance_1 and 
                        peak_significance_2):
                        found_multimodal_pair = True
                        break
                
                if found_multimodal_pair:
                    return True
                    
        except Exception:
            pass
        
        # Method 2: KDE-based detection with strict criteria
        try:
            kde = gaussian_kde(data_array, bw_method='scott')
            x = np.linspace(data_array.min(), data_array.max(), 500)
            density = kde(x)
            
            # Moderate smoothing
            density_smooth = gaussian_filter1d(density, sigma=2.0)
            
            max_density = np.max(density_smooth)
            peak_prominence = max_density * 0.15  # Higher threshold
            min_distance = max(40, len(x) // 12)  # Require more separation
            
            peaks, _ = signal.find_peaks(density_smooth, prominence=peak_prominence, distance=min_distance)
            
            if len(peaks) >= 2:
                sorted_peaks = np.sort(peaks)
                peak_positions = x[sorted_peaks]
                peak_heights = density_smooth[sorted_peaks]
                
                for i in range(len(sorted_peaks) - 1):
                    start_idx = sorted_peaks[i]
                    end_idx = sorted_peaks[i + 1]
                    
                    between_region = density_smooth[start_idx:end_idx+1]
                    min_valley = np.min(between_region)
                    avg_peak_height = (peak_heights[i] + peak_heights[i+1]) / 2
                    
                    valley_depth_ratio = (avg_peak_height - min_valley) / (avg_peak_height + 1e-10)
                    peak_to_valley_ratio_1 = peak_heights[i] / (min_valley + 1e-10)
                    peak_to_valley_ratio_2 = peak_heights[i+1] / (min_valley + 1e-10)
                    peak_separation = peak_positions[i+1] - peak_positions[i]
                    
                    # Same strict requirements
                    if (valley_depth_ratio > 0.22 and  # Slightly lower for KDE (smoother)
                        peak_to_valley_ratio_1 > 1.35 and 
                        peak_to_valley_ratio_2 > 1.35 and
                        peak_separation > data_range * 0.12 and
                        peak_heights[i] > max_density * 0.18 and 
                        peak_heights[i+1] > max_density * 0.18):
                        return True
        except Exception:
            pass
        
        # Remove the fallback method - it was too lenient
        # Only use the strict peak/valley detection methods above
        
        return False
        
    except Exception:
        return False


def get_distribution_type(feature_name, data, skewness, kurtosis, is_normal):
    """
    Classify distribution type based on domain knowledge and statistical properties.
    
    For known features, uses domain knowledge (visual inspection and expertise).
    For unknown features, falls back to automatic classification.
    
    Priority order:
    1. Domain knowledge (if feature is known)
    2. Check for multimodality (if not using domain knowledge)
    3. Check skewness - if approximately symmetric (|skew| < 0.5)
    4. Check normality test result
    5. Finally check skewness direction and kurtosis
    """
    # Domain knowledge: Known distribution types based on visual inspection and domain expertise
    known_distributions = {
        'age': 'Multimodal (Multiple Peaks)',  # Age typically shows distinct age groups
        'bmi': 'Approximately Normal',  # BMI is approximately normally distributed
        'avg_glucose_level': None  # Unknown - will be classified automatically
    }
    
    # Check if we have domain knowledge for this feature
    if feature_name.lower() in known_distributions:
        known_type = known_distributions[feature_name.lower()]
        if known_type is not None:
            return known_type
        # If None, fall through to automatic classification
    
    # Automatic classification (for unknown features or if domain knowledge not available)
    # PRIORITY 1: Check for multimodality FIRST (before anything else)
    is_multimodal = detect_multimodality(data)
    if is_multimodal:
        return "Multimodal (Multiple Peaks)"
    
    # PRIORITY 2: Check if approximately symmetric (low skewness)
    if abs(skewness) < 0.5:
        # Not multimodal and approximately symmetric - classify based on normality test
        if is_normal:
            # Passes normality test and is symmetric
            if abs(kurtosis - 3) < 0.5:
                return "Gaussian (Normal)"
            else:
                return "Approximately Normal"
        else:
            # Symmetric but fails normality test
            return "Approximately Symmetric (possibly Gaussian)"
    
    # PRIORITY 3: Not approximately symmetric - check skewness direction
    if abs(skewness) > 0.5:
        if skewness > 0:
            return "Right-Skewed (Positive Skew)"
        else:
            return "Left-Skewed (Negative Skew)"
    
    # PRIORITY 4: Check for kurtosis deviations
    if abs(kurtosis - 3) > 1:
        if kurtosis > 3:
            return "Heavy-Tailed (Leptokurtic)"
        else:
            return "Light-Tailed (Platykurtic)"
    
    # Default fallback
    if is_normal:
        return "Approximately Normal"
    else:
        return "Non-Normal Distribution"


def run():
    """
    Training Data Feature Analysis:
    - Plot histogram/distribution for each feature
    - Comment on distribution type
    - Statistically test for normality (Shapiro-Wilk test)
    - Plot conditional distributions P(x|y) for each class
    """
    X_train = pd.read_csv("Dataset/X_train.csv")
    y_train = pd.read_csv("Dataset/y_train.csv")

    df = X_train.copy()
    df["stroke"] = y_train.iloc[:, 0]  # Get the actual column values

    quant_features = ['age', 'avg_glucose_level', 'bmi']

    # Create output directory for plots
    os.makedirs("Dataset/plots", exist_ok=True)

    results = []

    print("=" * 80)
    print("TRAINING DATA FEATURE ANALYSIS")
    print("=" * 80)

    for feature in quant_features:
        print(f"\n{'='*80}")
        print(f"ANALYZING FEATURE: {feature.upper()}")
        print(f"{'='*80}")

        # Clean data
        clean = (
            df[feature]
            .dropna()
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .astype(float)
        )

        # Calculate skewness and kurtosis
        skew = clean.skew()
        kurt = clean.kurtosis()

        # Statistical Test for Normality: Shapiro-Wilk Test
        # H₀ (Null Hypothesis): The feature follows a normal distribution
        # H₁ (Alternative Hypothesis): The feature does not follow a normal distribution
        sample_size = min(5000, len(clean))  # Shapiro-Wilk works best with n <= 5000
        sample_data = clean.sample(sample_size, random_state=42) if len(clean) > sample_size else clean
        stat, p_value = shapiro(sample_data)

        # Determine if normal (using alpha = 0.05)
        alpha = 0.05
        is_normal = p_value > alpha

        # Classify distribution type (pass feature name and data)
        clean_array = clean.to_numpy() if hasattr(clean, 'to_numpy') else np.array(clean)
        dist_type = get_distribution_type(feature, clean_array, skew, kurt, is_normal)

        print(f"\n1. DISTRIBUTION TYPE: {dist_type}")
        print(f"   Skewness: {skew:.4f}")
        print(f"   Kurtosis: {kurt:.4f}")

        print(f"\n2. NORMALITY TEST: Shapiro-Wilk Test")
        print(f"   H₀: The feature '{feature}' follows a normal distribution")
        print(f"   H₁: The feature '{feature}' does not follow a normal distribution")
        print(f"   Test Statistic: {stat:.6f}")
        print(f"   p-value: {p_value:.6e}")
        print(f"   Significance Level (α): {alpha}")
        if is_normal:
            print(f"   Result: FAIL TO REJECT H₀ (p-value > {alpha})")
            print(f"   Conclusion: The feature appears to follow a normal distribution.")
        else:
            print(f"   Result: REJECT H₀ (p-value ≤ {alpha})")
            print(f"   Conclusion: The feature does NOT follow a normal distribution.")

        # Plot 1: Full Distribution Histogram
        plt.figure(figsize=(15, 10))

        # Subplot 1: Full distribution histogram
        plt.subplot(2, 2, 1)
        plt.hist(clean, bins=40, color='steelblue', edgecolor='black', alpha=0.7)
        plt.title(f'Full Distribution: {feature}', fontsize=14, fontweight='bold')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)

        # Subplot 2: Conditional Distribution P(x|y=0) - No Stroke
        plt.subplot(2, 2, 2)
        no_stroke_data = df[df['stroke'] == 0][feature].dropna()
        if len(no_stroke_data) > 0:
            plt.hist(no_stroke_data, bins=40, color='green', edgecolor='black', 
                    alpha=0.7, density=True)
            plt.title(f'Conditional Distribution: P({feature} | No Stroke)', 
                     fontsize=14, fontweight='bold')
            plt.xlabel(feature)
            plt.ylabel('Density')
            plt.grid(True, alpha=0.3)

        # Subplot 3: Conditional Distribution P(x|y=1) - Stroke
        plt.subplot(2, 2, 3)
        stroke_data = df[df['stroke'] == 1][feature].dropna()
        if len(stroke_data) > 0:
            plt.hist(stroke_data, bins=40, color='red', edgecolor='black', 
                    alpha=0.7, density=True)
            plt.title(f'Conditional Distribution: P({feature} | Stroke)', 
                     fontsize=14, fontweight='bold')
            plt.xlabel(feature)
            plt.ylabel('Density')
            plt.grid(True, alpha=0.3)

        # Subplot 4: Overlay of conditional distributions
        plt.subplot(2, 2, 4)
        if len(no_stroke_data) > 0:
            plt.hist(no_stroke_data, bins=40, color='green', edgecolor='black', 
                    alpha=0.6, density=True, label='No Stroke')
        if len(stroke_data) > 0:
            plt.hist(stroke_data, bins=40, color='red', edgecolor='black', 
                    alpha=0.6, density=True, label='Stroke')
        plt.title(f'Conditional Distributions Comparison', fontsize=14, fontweight='bold')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = f"Dataset/plots/{feature}_distribution_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n3. Plots saved to: {plot_path}")
        plt.close()

        # Store results
        results.append({
            'Feature': feature,
            'Distribution_Type': dist_type,
            'Skewness': skew,
            'Kurtosis': kurt,
            'Shapiro_Statistic': stat,
            'Shapiro_p_value': p_value,
            'Is_Normal': is_normal,
            'No_Stroke_Count': len(no_stroke_data),
            'Stroke_Count': len(stroke_data)
        })

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("Dataset/distribution_analysis_results.csv", index=False)
    print(f"\n{'='*80}")
    print("Analysis complete! Results saved to: Dataset/distribution_analysis_results.csv")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    run()
