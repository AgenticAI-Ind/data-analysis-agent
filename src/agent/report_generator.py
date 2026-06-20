"""
Report Generator - Automated report creation
Generates professional analysis reports in multiple formats.
"""

import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger


class ReportGenerator:
    """
    Automated report generation engine.

    Features:
    - Multiple output formats (PDF, HTML, Markdown)
    - Custom templates
    - Chart embedding
    - Executive summaries
    """

    def __init__(self):
        """Initialize report generator"""
        logger.info("ReportGenerator initialized")

    def generate(
        self,
        df: pd.DataFrame,
        title: str,
        sections: Optional[List[str]] = None,
        format: str = "markdown"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report.

        Args:
            df: DataFrame to analyze
            title: Report title
            sections: Sections to include
            format: Output format (markdown, html, pdf)

        Returns:
            Report content and metadata
        """
        logger.info(f"Generating {format} report: {title}")

        if sections is None:
            sections = ["summary", "statistics", "quality", "insights"]

        report_parts = []

        # Header
        report_parts.append(self._generate_header(title))

        # Sections
        if "summary" in sections:
            report_parts.append(self._generate_summary(df))

        if "statistics" in sections:
            report_parts.append(self._generate_statistics(df))

        if "quality" in sections:
            report_parts.append(self._generate_quality_section(df))

        if "insights" in sections:
            report_parts.append(self._generate_insights(df))

        # Footer
        report_parts.append(self._generate_footer())

        content = "\n\n".join(report_parts)

        return {
            "content": content,
            "format": format,
            "file_content": self._convert_format(content, format),
            "mime_type": self._get_mime_type(format)
        }

    def _generate_header(self, title: str) -> str:
        """Generate report header"""
        return f"""# {title}

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Report Type:** Data Analysis Report

---
"""

    def _generate_summary(self, df: pd.DataFrame) -> str:
        """Generate executive summary"""
        return f"""## Executive Summary

This report analyzes a dataset containing **{len(df):,}** records across **{len(df.columns)}** variables.

### Key Metrics:
- **Total Records:** {len(df):,}
- **Total Columns:** {len(df.columns)}
- **Date Range:** {self._get_date_range(df)}
- **Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
"""

    def _generate_statistics(self, df: pd.DataFrame) -> str:
        """Generate statistical summary"""
        stats = df.describe()

        return f"""## Statistical Summary

### Descriptive Statistics

{stats.to_markdown()}

### Data Types

{df.dtypes.to_frame('Type').to_markdown()}
"""

    def _generate_quality_section(self, df: pd.DataFrame) -> str:
        """Generate data quality section"""
        missing_data = df.isnull().sum()
        missing_pct = (missing_data / len(df) * 100).round(2)

        quality_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing %': missing_pct.values
        })

        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        quality_score = ((total_cells - missing_cells) / total_cells * 100)

        return f"""## Data Quality Assessment

### Overall Quality Score: {quality_score:.1f}%

### Missing Data Analysis

{quality_df[quality_df['Missing Count'] > 0].to_markdown(index=False)}

### Duplicate Records: {df.duplicated().sum():,}
"""

    def _generate_insights(self, df: pd.DataFrame) -> str:
        """Generate automated insights"""
        insights = []

        # Check for correlations
        numeric_df = df.select_dtypes(include=['number'])
        if len(numeric_df.columns) > 1:
            corr = numeric_df.corr()
            high_corr = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    if abs(corr.iloc[i, j]) > 0.7:
                        high_corr.append(
                            f"{corr.columns[i]} and {corr.columns[j]} ({corr.iloc[i, j]:.2f})"
                        )

            if high_corr:
                insights.append(f"**Strong Correlations Found:** {', '.join(high_corr[:3])}")

        # Check for skewness
        for col in numeric_df.columns:
            skew = numeric_df[col].skew()
            if abs(skew) > 1:
                insights.append(f"**{col}** is highly skewed (skewness: {skew:.2f})")

        # Check for outliers
        outlier_cols = []
        for col in numeric_df.columns:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((numeric_df[col] < (Q1 - 1.5 * IQR)) |
                       (numeric_df[col] > (Q3 + 1.5 * IQR))).sum()
            if outliers > 0:
                outlier_cols.append(f"{col} ({outliers} outliers)")

        if outlier_cols:
            insights.append(f"**Outliers detected in:** {', '.join(outlier_cols[:3])}")

        insights_text = "\n".join([f"- {insight}" for insight in insights])

        return f"""## Key Insights

{insights_text if insights else "No significant patterns detected."}

## Recommendations

- Verify data quality for columns with high missing rates
- Investigate outliers before statistical analysis
- Consider data transformation for highly skewed variables
- Review correlations for potential multicollinearity
"""

    def _generate_footer(self) -> str:
        """Generate report footer"""
        return """
---

**Report generated by Data Analysis Agent**
*Powered by AI and statistical analysis*
"""

    def _get_date_range(self, df: pd.DataFrame) -> str:
        """Get date range from DataFrame"""
        date_cols = df.select_dtypes(include=['datetime64']).columns

        if len(date_cols) > 0:
            col = date_cols[0]
            return f"{df[col].min()} to {df[col].max()}"

        return "N/A"

    def _convert_format(self, content: str, format: str) -> bytes:
        """Convert content to target format"""
        if format == "markdown":
            return content.encode('utf-8')
        elif format == "html":
            # Simple HTML conversion
            html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Data Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
    </style>
</head>
<body>
{content.replace('# ', '<h1>').replace('## ', '<h2>').replace('### ', '<h3>')}
</body>
</html>"""
            return html.encode('utf-8')
        else:
            return content.encode('utf-8')

    def _get_mime_type(self, format: str) -> str:
        """Get MIME type for format"""
        mime_types = {
            "markdown": "text/markdown",
            "html": "text/html",
            "pdf": "application/pdf"
        }
        return mime_types.get(format, "text/plain")


# Example usage
if __name__ == "__main__":
    import numpy as np

    # Sample data
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100),
        'sales': np.random.randint(100, 1000, 100),
        'customers': np.random.randint(10, 50, 100),
        'region': ['North', 'South', 'East', 'West'] * 25
    })

    generator = ReportGenerator()

    report = generator.generate(
        df=df,
        title="Q1 Sales Analysis",
        sections=["summary", "statistics", "quality", "insights"],
        format="markdown"
    )

    print(report['content'])
