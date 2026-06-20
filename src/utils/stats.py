"""
Statistical Utilities
Common statistical functions and analyses.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Tuple
from loguru import logger


def calculate_basic_stats(series: pd.Series) -> Dict[str, Any]:
    """
    Calculate basic statistics for a series.

    Args:
        series: Pandas Series

    Returns:
        Dictionary with statistical metrics
    """
    return {
        "count": len(series),
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "min": series.min(),
        "max": series.max(),
        "q25": series.quantile(0.25),
        "q75": series.quantile(0.75),
        "skewness": series.skew(),
        "kurtosis": series.kurtosis()
    }


def detect_outliers_iqr(
    series: pd.Series,
    multiplier: float = 1.5
) -> Tuple[List[int], float, float]:
    """
    Detect outliers using IQR method.

    Args:
        series: Pandas Series
        multiplier: IQR multiplier (default 1.5)

    Returns:
        Tuple of (outlier_indices, lower_bound, upper_bound)
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    outliers = series[(series < lower_bound) | (series > upper_bound)].index.tolist()

    return outliers, lower_bound, upper_bound


def detect_outliers_zscore(
    series: pd.Series,
    threshold: float = 3.0
) -> List[int]:
    """
    Detect outliers using Z-score method.

    Args:
        series: Pandas Series
        threshold: Z-score threshold (default 3.0)

    Returns:
        List of outlier indices
    """
    z_scores = np.abs(stats.zscore(series.dropna()))
    outlier_mask = z_scores > threshold

    return series.dropna()[outlier_mask].index.tolist()


def test_normality(series: pd.Series) -> Dict[str, Any]:
    """
    Test if series follows normal distribution.

    Args:
        series: Pandas Series

    Returns:
        Dictionary with test results
    """
    # Shapiro-Wilk test
    statistic, p_value = stats.shapiro(series.dropna())

    return {
        "test": "Shapiro-Wilk",
        "statistic": statistic,
        "p_value": p_value,
        "is_normal": p_value > 0.05,
        "interpretation": "Data appears normally distributed" if p_value > 0.05 else "Data does not appear normally distributed"
    }


def calculate_correlation(
    df: pd.DataFrame,
    method: str = 'pearson'
) -> pd.DataFrame:
    """
    Calculate correlation matrix.

    Args:
        df: DataFrame with numeric columns
        method: Correlation method (pearson, spearman, kendall)

    Returns:
        Correlation matrix
    """
    numeric_df = df.select_dtypes(include=['number'])
    return numeric_df.corr(method=method)


def find_strong_correlations(
    df: pd.DataFrame,
    threshold: float = 0.7,
    method: str = 'pearson'
) -> List[Tuple[str, str, float]]:
    """
    Find pairs of strongly correlated variables.

    Args:
        df: DataFrame
        threshold: Correlation threshold
        method: Correlation method

    Returns:
        List of (var1, var2, correlation) tuples
    """
    corr_matrix = calculate_correlation(df, method)

    strong_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) >= threshold:
                strong_corr.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_value
                ))

    return sorted(strong_corr, key=lambda x: abs(x[2]), reverse=True)


def perform_ttest(
    group1: pd.Series,
    group2: pd.Series,
    equal_var: bool = True
) -> Dict[str, Any]:
    """
    Perform independent t-test.

    Args:
        group1: First group
        group2: Second group
        equal_var: Assume equal variance

    Returns:
        Test results
    """
    statistic, p_value = stats.ttest_ind(
        group1.dropna(),
        group2.dropna(),
        equal_var=equal_var
    )

    return {
        "test": "Independent T-Test",
        "statistic": statistic,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "interpretation": "Groups are significantly different" if p_value < 0.05 else "No significant difference between groups"
    }


def calculate_confidence_interval(
    series: pd.Series,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence interval for mean.

    Args:
        series: Pandas Series
        confidence: Confidence level

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    mean = series.mean()
    sem = stats.sem(series.dropna())
    margin = sem * stats.t.ppf((1 + confidence) / 2, len(series) - 1)

    return (mean - margin, mean + margin)


def calculate_percentiles(
    series: pd.Series,
    percentiles: List[float] = None
) -> Dict[float, float]:
    """
    Calculate custom percentiles.

    Args:
        series: Pandas Series
        percentiles: List of percentile values (0-100)

    Returns:
        Dictionary mapping percentiles to values
    """
    if percentiles is None:
        percentiles = [10, 25, 50, 75, 90, 95, 99]

    return {
        p: series.quantile(p / 100)
        for p in percentiles
    }


def perform_chi_square_test(
    observed: np.ndarray,
    expected: np.ndarray = None
) -> Dict[str, Any]:
    """
    Perform chi-square test.

    Args:
        observed: Observed frequencies
        expected: Expected frequencies (optional)

    Returns:
        Test results
    """
    if expected is None:
        statistic, p_value = stats.chisquare(observed)
    else:
        statistic, p_value = stats.chisquare(observed, expected)

    return {
        "test": "Chi-Square",
        "statistic": statistic,
        "p_value": p_value,
        "significant": p_value < 0.05
    }


def calculate_moving_average(
    series: pd.Series,
    window: int = 7
) -> pd.Series:
    """
    Calculate moving average.

    Args:
        series: Time series data
        window: Window size

    Returns:
        Moving average series
    """
    return series.rolling(window=window).mean()


def calculate_exponential_smoothing(
    series: pd.Series,
    alpha: float = 0.3
) -> pd.Series:
    """
    Calculate exponentially weighted moving average.

    Args:
        series: Time series data
        alpha: Smoothing factor

    Returns:
        Smoothed series
    """
    return series.ewm(alpha=alpha, adjust=False).mean()


def detect_trends(series: pd.Series) -> Dict[str, Any]:
    """
    Detect trends in time series.

    Args:
        series: Time series data

    Returns:
        Trend analysis results
    """
    # Linear regression for trend
    x = np.arange(len(series))
    y = series.values

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    trend_type = "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"

    return {
        "trend": trend_type,
        "slope": slope,
        "r_squared": r_value ** 2,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "interpretation": f"Significant {trend_type} trend detected" if p_value < 0.05 else "No significant trend"
    }


# Example usage
if __name__ == "__main__":
    # Sample data
    data = pd.Series(np.random.normal(100, 15, 1000))

    # Basic stats
    stats_result = calculate_basic_stats(data)
    print("Basic Statistics:")
    for key, value in stats_result.items():
        print(f"  {key}: {value:.2f}")

    # Outliers
    outliers, lower, upper = detect_outliers_iqr(data)
    print(f"\nOutliers detected: {len(outliers)}")
    print(f"Bounds: [{lower:.2f}, {upper:.2f}]")

    # Normality test
    normality = test_normality(data)
    print(f"\nNormality: {normality['interpretation']}")
    print(f"P-value: {normality['p_value']:.4f}")
