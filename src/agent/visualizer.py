"""
Data Visualizer - Chart generation engine
Creates interactive visualizations with Plotly.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional
from loguru import logger


class DataVisualizer:
    """
    Interactive visualization engine using Plotly.

    Features:
    - Multiple chart types
    - Auto chart type selection
    - Interactive elements
    - Export capabilities
    """

    def __init__(self):
        """Initialize visualizer"""
        logger.info("DataVisualizer initialized")

    def create_chart(
        self,
        df: pd.DataFrame,
        chart_type: str,
        x: str,
        y: Optional[str] = None,
        color: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """
        Create interactive chart.

        Args:
            df: DataFrame with data
            chart_type: Type of chart (line_chart, bar_chart, etc.)
            x: X-axis column
            y: Y-axis column (optional for some charts)
            color: Color grouping column
            **kwargs: Additional chart parameters

        Returns:
            Plotly Figure object
        """
        logger.info(f"Creating {chart_type} chart")

        if chart_type == "line_chart":
            return self._create_line_chart(df, x, y, color, **kwargs)
        elif chart_type == "bar_chart":
            return self._create_bar_chart(df, x, y, color, **kwargs)
        elif chart_type == "scatter_plot":
            return self._create_scatter(df, x, y, color, **kwargs)
        elif chart_type == "histogram":
            return self._create_histogram(df, x, **kwargs)
        elif chart_type == "box_plot":
            return self._create_box_plot(df, x, y, **kwargs)
        elif chart_type == "pie_chart":
            return self._create_pie_chart(df, x, y, **kwargs)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")

    def _create_line_chart(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        color: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create line chart"""
        fig = px.line(
            df,
            x=x,
            y=y,
            color=color,
            title=kwargs.get('title', f'{y} over {x}'),
            template='plotly_dark'
        )
        fig.update_layout(hovermode='x unified')
        return fig

    def _create_bar_chart(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        color: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create bar chart"""
        fig = px.bar(
            df,
            x=x,
            y=y,
            color=color,
            title=kwargs.get('title', f'{y} by {x}'),
            template='plotly_dark'
        )
        return fig

    def _create_scatter(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        color: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create scatter plot"""
        fig = px.scatter(
            df,
            x=x,
            y=y,
            color=color,
            title=kwargs.get('title', f'{y} vs {x}'),
            template='plotly_dark',
            trendline='ols' if kwargs.get('trendline') else None
        )
        return fig

    def _create_histogram(
        self,
        df: pd.DataFrame,
        x: str,
        **kwargs
    ) -> go.Figure:
        """Create histogram"""
        fig = px.histogram(
            df,
            x=x,
            title=kwargs.get('title', f'Distribution of {x}'),
            template='plotly_dark',
            nbins=kwargs.get('bins', 30)
        )
        return fig

    def _create_box_plot(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        **kwargs
    ) -> go.Figure:
        """Create box plot"""
        fig = px.box(
            df,
            x=x,
            y=y,
            title=kwargs.get('title', f'{y} Distribution by {x}'),
            template='plotly_dark'
        )
        return fig

    def _create_pie_chart(
        self,
        df: pd.DataFrame,
        names: str,
        values: str,
        **kwargs
    ) -> go.Figure:
        """Create pie chart"""
        fig = px.pie(
            df,
            names=names,
            values=values,
            title=kwargs.get('title', f'{values} by {names}'),
            template='plotly_dark'
        )
        return fig

    def create_correlation_heatmap(
        self,
        df: pd.DataFrame,
        **kwargs
    ) -> go.Figure:
        """
        Create correlation heatmap for numeric columns.

        Args:
            df: DataFrame with numeric data
            **kwargs: Additional parameters

        Returns:
            Plotly Figure
        """
        logger.info("Creating correlation heatmap")

        numeric_df = df.select_dtypes(include=['number'])
        corr_matrix = numeric_df.corr()

        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title=kwargs.get('title', 'Correlation Heatmap'),
            template='plotly_dark',
            color_continuous_scale='RdBu_r'
        )

        return fig


# Example usage
if __name__ == "__main__":
    import numpy as np

    # Sample data
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100),
        'sales': np.random.randint(100, 1000, 100),
        'region': ['North', 'South', 'East', 'West'] * 25
    })

    visualizer = DataVisualizer()

    # Create line chart
    fig = visualizer.create_chart(
        df=df,
        chart_type='line_chart',
        x='date',
        y='sales',
        color='region'
    )

    fig.show()
