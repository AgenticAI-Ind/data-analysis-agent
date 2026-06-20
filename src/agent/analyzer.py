"""
Data Analyzer - Core analysis engine
Provides statistical analysis and AI-powered insights.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from scipy import stats
from langchain_community.llms import Ollama
from loguru import logger


class DataAnalyzer:
    """
    Core data analysis engine with AI capabilities.

    Features:
    - Statistical summarization
    - Data quality checks
    - Anomaly detection
    - AI-generated insights
    - Natural language queries
    """

    def __init__(self, model: str = "llama3.2"):
        """
        Initialize data analyzer.

        Args:
            model: Ollama model for AI insights
        """
        self.llm = Ollama(model=model, temperature=0.3)
        logger.info(f"DataAnalyzer initialized with {model}")

    def get_statistical_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get comprehensive statistical summary.

        Args:
            df: DataFrame to analyze

        Returns:
            Summary statistics DataFrame
        """
        logger.info(f"Generating statistical summary for {len(df)} rows")
        return df.describe(include='all')

    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check data quality metrics.

        Args:
            df: DataFrame to check

        Returns:
            Dictionary with quality metrics
        """
        logger.info("Checking data quality")

        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        complete_rows = df.dropna().shape[0]
        duplicate_rows = df.duplicated().sum()

        columns_with_issues = 0
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                columns_with_issues += 1

        quality_score = ((total_cells - missing_cells) / total_cells * 100) if total_cells > 0 else 0

        return {
            "overall_score": round(quality_score, 1),
            "total_rows": len(df),
            "complete_rows": complete_rows,
            "complete_rows_pct": round((complete_rows / len(df) * 100) if len(df) > 0 else 0, 1),
            "missing_cells": missing_cells,
            "duplicate_count": duplicate_rows,
            "columns_with_issues": columns_with_issues
        }

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        confidence: float = 0.95
    ) -> Dict[str, List[int]]:
        """
        Detect anomalies using statistical methods.

        Args:
            df: DataFrame to analyze
            confidence: Confidence level for outlier detection

        Returns:
            Dictionary mapping column names to anomaly indices
        """
        logger.info(f"Detecting anomalies with {confidence} confidence")

        anomalies = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if df[col].notna().sum() < 3:
                continue

            # Z-score method
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            threshold = stats.norm.ppf((1 + confidence) / 2)

            outlier_indices = df[col].dropna()[z_scores > threshold].index.tolist()

            if outlier_indices:
                anomalies[col] = outlier_indices

        return anomalies

    def generate_insights(self, df: pd.DataFrame, model: str = None) -> str:
        """
        Generate AI-powered insights about the data.

        Args:
            df: DataFrame to analyze
            model: Optional model override

        Returns:
            AI-generated insights as markdown
        """
        logger.info("Generating AI insights")

        # Prepare data summary for LLM
        summary = f"""Dataset Overview:
- Rows: {len(df)}
- Columns: {len(df.columns)}
- Column Names: {', '.join(df.columns.tolist())}

Statistical Summary:
{df.describe().to_string()}

Data Types:
{df.dtypes.to_string()}
"""

        prompt = f"""Analyze this dataset and provide 3-5 key insights:

{summary[:1500]}

Provide insights about:
1. Overall data patterns
2. Notable relationships
3. Potential data quality issues
4. Recommendations for analysis

Insights:
"""

        try:
            insights = self.llm(prompt)
            return insights
        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
            return "Unable to generate AI insights at this time."

    def answer_question(
        self,
        df: pd.DataFrame,
        question: str,
        model: str = None
    ) -> Dict[str, Any]:
        """
        Answer natural language questions about the data.

        Args:
            df: DataFrame to query
            question: Natural language question
            model: Optional model override

        Returns:
            Dictionary with answer, data, and visualization info
        """
        logger.info(f"Answering question: {question}")

        # Prepare context
        context = f"""Dataset Information:
Columns: {', '.join(df.columns.tolist())}
Rows: {len(df)}
Sample Data:
{df.head(3).to_string()}

Question: {question}

Based on this data, provide:
1. A clear answer to the question
2. Relevant statistics
3. Suggestions for visualization
"""

        try:
            answer = self.llm(context[:2000])

            return {
                "answer": answer,
                "data": None,  # Could include filtered DataFrame
                "visualization": None  # Visualization suggestion
            }
        except Exception as e:
            logger.error(f"Failed to answer question: {e}")
            return {
                "answer": "Unable to process question",
                "data": None,
                "visualization": None
            }

    def generate_report(
        self,
        df: pd.DataFrame,
        title: str = "Data Analysis Report",
        include_visuals: bool = True,
        format: str = "markdown"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report.

        Args:
            df: DataFrame to analyze
            title: Report title
            include_visuals: Include visualization recommendations
            format: Output format (markdown, html, pdf)

        Returns:
            Dictionary with report content and metadata
        """
        logger.info(f"Generating {format} report: {title}")

        # Generate report sections
        sections = []

        sections.append(f"# {title}\n")
        sections.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
        sections.append(f"**Dataset:** {len(df)} rows × {len(df.columns)} columns\n")

        # Summary statistics
        sections.append("\n## Summary Statistics\n")
        sections.append(df.describe().to_markdown())

        # Data quality
        quality = self.check_data_quality(df)
        sections.append("\n## Data Quality\n")
        sections.append(f"- Overall Score: {quality['overall_score']}%\n")
        sections.append(f"- Complete Rows: {quality['complete_rows_pct']}%\n")
        sections.append(f"- Duplicates: {quality['duplicate_count']}\n")

        # AI Insights
        insights = self.generate_insights(df)
        sections.append("\n## Key Insights\n")
        sections.append(insights)

        content = '\n'.join(sections)

        return {
            "content": content,
            "format": format,
            "file_content": content.encode('utf-8'),
            "mime_type": "text/markdown"
        }


# Example usage
if __name__ == "__main__":
    # Create sample data
    df = pd.DataFrame({
        'product': ['A', 'B', 'C', 'D', 'E'] * 20,
        'sales': np.random.randint(100, 1000, 100),
        'price': np.random.uniform(10, 100, 100),
        'region': ['North', 'South', 'East', 'West'] * 25
    })

    analyzer = DataAnalyzer()

    # Get summary
    print("Statistical Summary:")
    print(analyzer.get_statistical_summary(df))

    # Check quality
    quality = analyzer.check_data_quality(df)
    print(f"\nData Quality Score: {quality['overall_score']}%")

    # Detect anomalies
    anomalies = analyzer.detect_anomalies(df)
    print(f"\nAnomalies detected: {len(anomalies)} columns")
