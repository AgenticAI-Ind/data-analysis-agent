"""
Streamlit UI for Data Analysis Agent
Interactive web interface for data analysis and visualization.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.agent.analyzer import DataAnalyzer
from src.agent.visualizer import DataVisualizer
from src.agent.sql_generator import SQLGenerator


# Page configuration
st.set_page_config(
    page_title="Data Analysis Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = DataAnalyzer()
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = DataVisualizer()
    if 'sql_generator' not in st.session_state:
        st.session_state.sql_generator = SQLGenerator()
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []


def main():
    """Main application"""
    initialize_session_state()

    # Header
    st.markdown('<h1 class="main-header">📊 Data Analysis Agent</h1>', unsafe_allow_html=True)
    st.markdown("### AI-powered data analysis and insights")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Model selection
        model = st.selectbox(
            "LLM Model",
            ["llama3.2", "mistral", "codellama"],
            help="Select the model for analysis"
        )

        # Analysis settings
        st.subheader("Analysis Settings")
        auto_detect_anomalies = st.checkbox("Auto-detect anomalies", value=True)
        confidence_level = st.slider("Confidence Level", 0.8, 0.99, 0.95)

        st.divider()

        # Sample datasets
        st.subheader("📁 Sample Datasets")
        if st.button("Load Sales Dataset"):
            st.session_state.df = load_sample_data("sales")
            st.success("Sample data loaded!")

    # Main content area
    tabs = st.tabs([
        "📂 Upload Data",
        "🔍 Analyze",
        "📊 Visualize",
        "💬 Ask Questions",
        "📝 Generate Report"
    ])

    # Tab 1: Upload Data
    with tabs[0]:
        st.header("Upload Your Data")

        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            help="Upload your data file for analysis"
        )

        if uploaded_file:
            try:
                # Load data
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                st.session_state.df = df
                st.success(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")

                # Preview data
                st.subheader("Data Preview")
                st.dataframe(df.head(10), use_container_width=True)

                # Basic statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Rows", f"{len(df):,}")
                with col2:
                    st.metric("Total Columns", len(df.columns))
                with col3:
                    st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                with col4:
                    missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
                    st.metric("Missing Values", f"{missing_pct:.1f}%")

            except Exception as e:
                st.error(f"Error loading file: {e}")

    # Tab 2: Analyze
    with tabs[1]:
        st.header("Automated Analysis")

        if st.session_state.df is not None:
            df = st.session_state.df

            if st.button("🚀 Run Full Analysis", type="primary"):
                with st.spinner("Analyzing data..."):
                    analyzer = st.session_state.analyzer

                    # Statistical summary
                    st.subheader("📊 Statistical Summary")
                    summary = analyzer.get_statistical_summary(df)
                    st.dataframe(summary, use_container_width=True)

                    # Data quality check
                    st.subheader("✅ Data Quality")
                    quality = analyzer.check_data_quality(df)
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Quality Score", f"{quality['overall_score']:.1f}%")
                        st.metric("Complete Rows", f"{quality['complete_rows_pct']:.1f}%")

                    with col2:
                        st.metric("Duplicate Rows", quality['duplicate_count'])
                        st.metric("Columns with Issues", quality['columns_with_issues'])

                    # Correlations
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 1:
                        st.subheader("🔗 Correlations")
                        corr_matrix = df[numeric_cols].corr()

                        fig = px.imshow(
                            corr_matrix,
                            text_auto=True,
                            aspect="auto",
                            color_continuous_scale="RdBu_r",
                            title="Correlation Heatmap"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Anomalies
                    if auto_detect_anomalies:
                        st.subheader("⚠️ Anomaly Detection")
                        anomalies = analyzer.detect_anomalies(df, confidence=confidence_level)

                        if anomalies:
                            for col, outliers in anomalies.items():
                                st.warning(f"**{col}**: {len(outliers)} anomalies detected")
                                with st.expander(f"View {col} anomalies"):
                                    st.dataframe(df.iloc[outliers])
                        else:
                            st.success("No anomalies detected!")

                    # AI Insights
                    st.subheader("🤖 AI-Generated Insights")
                    with st.spinner("Generating insights..."):
                        insights = analyzer.generate_insights(df, model=model)
                        st.markdown(insights)

        else:
            st.info("👆 Please upload data in the Upload Data tab first")

    # Tab 3: Visualize
    with tabs[2]:
        st.header("Data Visualization")

        if st.session_state.df is not None:
            df = st.session_state.df
            visualizer = st.session_state.visualizer

            # Chart type selection
            chart_type = st.selectbox(
                "Select Chart Type",
                ["Line Chart", "Bar Chart", "Scatter Plot", "Histogram", "Box Plot", "Pie Chart"]
            )

            col1, col2 = st.columns(2)

            with col1:
                x_axis = st.selectbox("X-Axis", df.columns)

            with col2:
                if chart_type in ["Line Chart", "Bar Chart", "Scatter Plot"]:
                    y_axis = st.selectbox("Y-Axis", df.columns)
                else:
                    y_axis = None

            if st.button("Generate Chart"):
                try:
                    fig = visualizer.create_chart(
                        df=df,
                        chart_type=chart_type.lower().replace(" ", "_"),
                        x=x_axis,
                        y=y_axis
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating chart: {e}")

        else:
            st.info("👆 Please upload data first")

    # Tab 4: Ask Questions
    with tabs[3]:
        st.header("Ask Questions About Your Data")

        if st.session_state.df is not None:
            question = st.text_input(
                "What would you like to know?",
                placeholder="e.g., What are the top 5 products by revenue?"
            )

            if st.button("Ask", type="primary") and question:
                with st.spinner("Thinking..."):
                    analyzer = st.session_state.analyzer

                    # Get answer
                    result = analyzer.answer_question(
                        df=st.session_state.df,
                        question=question,
                        model=model
                    )

                    # Display answer
                    st.subheader("💡 Answer")
                    st.markdown(result['answer'])

                    # Display visualization if available
                    if result.get('visualization'):
                        st.subheader("📊 Visualization")
                        st.plotly_chart(result['visualization'], use_container_width=True)

                    # Display relevant data
                    if result.get('data'):
                        st.subheader("📋 Relevant Data")
                        st.dataframe(result['data'], use_container_width=True)

                    # Add to history
                    st.session_state.analysis_history.append({
                        'question': question,
                        'answer': result['answer']
                    })

            # Show history
            if st.session_state.analysis_history:
                st.divider()
                st.subheader("📜 Question History")
                for i, item in enumerate(reversed(st.session_state.analysis_history[-5:]), 1):
                    with st.expander(f"Q{i}: {item['question'][:50]}..."):
                        st.markdown(f"**Q:** {item['question']}")
                        st.markdown(f"**A:** {item['answer']}")

        else:
            st.info("👆 Please upload data first")

    # Tab 5: Generate Report
    with tabs[4]:
        st.header("Generate Analysis Report")

        if st.session_state.df is not None:
            report_title = st.text_input("Report Title", "Data Analysis Report")
            include_visuals = st.checkbox("Include Visualizations", value=True)
            report_format = st.radio("Format", ["PDF", "HTML", "Markdown"])

            if st.button("📄 Generate Report", type="primary"):
                with st.spinner("Generating report..."):
                    analyzer = st.session_state.analyzer

                    report = analyzer.generate_report(
                        df=st.session_state.df,
                        title=report_title,
                        include_visuals=include_visuals,
                        format=report_format.lower()
                    )

                    # Preview
                    st.subheader("Report Preview")
                    st.markdown(report['content'])

                    # Download button
                    st.download_button(
                        label=f"⬇️ Download {report_format}",
                        data=report['file_content'],
                        file_name=f"report.{report_format.lower()}",
                        mime=report['mime_type']
                    )

        else:
            st.info("👆 Please upload data first")


def load_sample_data(dataset_name):
    """Load sample datasets for demo"""
    if dataset_name == "sales":
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100),
            'product': ['A', 'B', 'C', 'D', 'E'] * 20,
            'quantity': pd.np.random.randint(1, 100, 100),
            'revenue': pd.np.random.uniform(100, 1000, 100),
            'region': ['North', 'South', 'East', 'West'] * 25
        })
    return None


if __name__ == "__main__":
    main()
