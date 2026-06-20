"""
Reusable UI Components for Streamlit
Common UI patterns and widgets.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Any, Dict, List, Optional


def render_metric_card(
    label: str,
    value: Any,
    delta: Optional[str] = None,
    help_text: Optional[str] = None
):
    """
    Render a metric card with optional delta.

    Args:
        label: Metric label
        value: Metric value
        delta: Optional change indicator
        help_text: Optional help tooltip
    """
    st.metric(
        label=label,
        value=value,
        delta=delta,
        help=help_text
    )


def render_data_preview(
    df: pd.DataFrame,
    title: str = "Data Preview",
    num_rows: int = 10
):
    """
    Render data preview table.

    Args:
        df: DataFrame to display
        title: Section title
        num_rows: Number of rows to show
    """
    st.subheader(title)
    st.dataframe(df.head(num_rows), use_container_width=True)

    with st.expander("Show full dataset"):
        st.dataframe(df, use_container_width=True)


def render_summary_stats(df: pd.DataFrame):
    """
    Render summary statistics cards.

    Args:
        df: DataFrame to summarize
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Rows", f"{len(df):,}")

    with col2:
        st.metric("Total Columns", len(df.columns))

    with col3:
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Memory", f"{memory_mb:.2f} MB")

    with col4:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
        st.metric("Missing", f"{missing_pct:.1f}%")


def render_filter_sidebar(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Render filter controls in sidebar.

    Args:
        df: DataFrame to filter
        columns: Columns to provide filters for

    Returns:
        Filtered DataFrame
    """
    st.sidebar.header("🔍 Filters")

    filtered_df = df.copy()

    if columns is None:
        columns = df.columns.tolist()

    for col in columns:
        if df[col].dtype in ['int64', 'float64']:
            # Numeric filter
            min_val = float(df[col].min())
            max_val = float(df[col].max())

            values = st.sidebar.slider(
                f"{col}",
                min_val,
                max_val,
                (min_val, max_val)
            )

            filtered_df = filtered_df[
                (filtered_df[col] >= values[0]) &
                (filtered_df[col] <= values[1])
            ]

        elif df[col].dtype == 'object':
            # Categorical filter
            unique_values = df[col].unique().tolist()

            if len(unique_values) <= 20:  # Only show for reasonable number of categories
                selected = st.sidebar.multiselect(
                    f"{col}",
                    unique_values,
                    default=unique_values
                )

                filtered_df = filtered_df[filtered_df[col].isin(selected)]

    return filtered_df


def render_chart_selector(
    df: pd.DataFrame,
    chart_types: List[str] = None
) -> Dict[str, Any]:
    """
    Render chart type and column selectors.

    Args:
        df: DataFrame for column options
        chart_types: Available chart types

    Returns:
        Dictionary with chart configuration
    """
    if chart_types is None:
        chart_types = ["Line Chart", "Bar Chart", "Scatter Plot", "Histogram"]

    col1, col2, col3 = st.columns(3)

    with col1:
        chart_type = st.selectbox("Chart Type", chart_types)

    with col2:
        x_axis = st.selectbox("X-Axis", df.columns)

    with col3:
        if chart_type in ["Line Chart", "Bar Chart", "Scatter Plot"]:
            y_axis = st.selectbox("Y-Axis", df.columns)
        else:
            y_axis = None

    return {
        "chart_type": chart_type,
        "x_axis": x_axis,
        "y_axis": y_axis
    }


def render_download_button(
    data: Any,
    filename: str,
    label: str = "Download",
    mime_type: str = "text/csv"
):
    """
    Render download button.

    Args:
        data: Data to download
        filename: Downloaded filename
        label: Button label
        mime_type: File MIME type
    """
    st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime=mime_type
    )


def render_correlation_matrix(df: pd.DataFrame):
    """
    Render correlation heatmap for numeric columns.

    Args:
        df: DataFrame to analyze
    """
    numeric_df = df.select_dtypes(include=['number'])

    if len(numeric_df.columns) < 2:
        st.warning("Need at least 2 numeric columns for correlation analysis")
        return

    st.subheader("📊 Correlation Matrix")

    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Feature Correlations"
    )

    st.plotly_chart(fig, use_container_width=True)


def render_missing_data_chart(df: pd.DataFrame):
    """
    Render missing data visualization.

    Args:
        df: DataFrame to analyze
    """
    st.subheader("🔍 Missing Data Analysis")

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing Count': missing.values,
        'Missing %': missing_pct.values
    })

    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(
        'Missing Count',
        ascending=False
    )

    if len(missing_df) > 0:
        fig = px.bar(
            missing_df,
            x='Column',
            y='Missing %',
            title='Missing Data by Column',
            labels={'Missing %': 'Missing Percentage (%)'}
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("✅ No missing data found!")


def render_distribution_chart(df: pd.DataFrame, column: str):
    """
    Render distribution chart for a column.

    Args:
        df: DataFrame
        column: Column to visualize
    """
    st.subheader(f"Distribution of {column}")

    if df[column].dtype in ['int64', 'float64']:
        fig = px.histogram(
            df,
            x=column,
            title=f'{column} Distribution',
            marginal='box'
        )
    else:
        value_counts = df[column].value_counts().head(10)
        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            labels={'x': column, 'y': 'Count'},
            title=f'Top 10 {column} Values'
        )

    st.plotly_chart(fig, use_container_width=True)


def render_success_message(message: str):
    """Render success message"""
    st.success(f"✅ {message}")


def render_error_message(message: str):
    """Render error message"""
    st.error(f"❌ {message}")


def render_info_message(message: str):
    """Render info message"""
    st.info(f"ℹ️ {message}")


def render_warning_message(message: str):
    """Render warning message"""
    st.warning(f"⚠️ {message}")


# Example usage
if __name__ == "__main__":
    st.set_page_config(page_title="Component Demo", layout="wide")

    st.title("Reusable Components Demo")

    # Sample data
    df = pd.DataFrame({
        'A': range(100),
        'B': range(100, 200),
        'C': ['X', 'Y', 'Z'] * 33 + ['X']
    })

    # Demo metrics
    st.header("Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card("Total Sales", "$1.2M", "+12%")
    with col2:
        render_metric_card("Customers", "1,234", "+5%")
    with col3:
        render_metric_card("Conversion", "3.4%", "-0.2%")

    # Demo data preview
    render_data_preview(df, "Sample Data")

    # Demo summary stats
    st.header("Summary Statistics")
    render_summary_stats(df)
