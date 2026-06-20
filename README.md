# Data Analysis & Insights Agent

A production-ready AI agent for automated data analysis, visualization, and business intelligence using Streamlit, Pandas, and LLMs.

## Features

- 📊 **Automated Data Analysis** - Upload CSV/Excel and get instant insights
- 📈 **Interactive Visualizations** - Dynamic charts with Plotly
- 🤖 **Natural Language Queries** - Ask questions in plain English
- 🔍 **Anomaly Detection** - Automatic outlier and trend identification
- 📝 **Report Generation** - Auto-generate executive summaries
- 🗄️ **SQL Query Generation** - Natural language to SQL conversion

## Tech Stack

- **Streamlit** - Interactive web interface
- **Pandas & NumPy** - Data manipulation
- **Plotly** - Interactive visualizations
- **Ollama** - Local LLM for analysis
- **SQLite/PostgreSQL** - Database connectivity
- **scikit-learn** - Statistical analysis and ML
- **Prophet** - Time series forecasting

## Architecture

```
data-analysis-agent/
├── src/
│   ├── agent/
│   │   ├── analyzer.py          # Core analysis logic
│   │   ├── visualizer.py        # Chart generation
│   │   ├── sql_generator.py     # NL to SQL conversion
│   │   └── report_generator.py  # Report creation
│   ├── ui/
│   │   ├── app.py               # Main Streamlit app
│   │   └── components.py        # Reusable UI components
│   └── utils/
│       ├── data_loader.py       # Data ingestion
│       └── stats.py             # Statistical utilities
├── data/
│   └── sample_datasets/
├── outputs/
│   └── reports/
├── requirements.txt
└── docker-compose.yml
```

## Installation

### Prerequisites

- Python 3.10+
- Ollama installed ([ollama.ai](https://ollama.ai))

### Setup

```bash
cd data-analysis-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Pull Ollama model
ollama pull llama3.2

# Run the app
streamlit run src/ui/app.py
```

## Usage

### Web Interface

1. **Upload Data**: Drag and drop CSV/Excel files
2. **Auto Analysis**: Get instant statistical summary
3. **Ask Questions**: Type natural language queries
4. **Generate Visualizations**: Create interactive charts
5. **Export Reports**: Download PDF/HTML reports

### Example Queries

```
"Show me the top 10 customers by revenue"
"What's the trend in sales over the last 6 months?"
"Identify any outliers in the price column"
"Compare category performance year over year"
"Generate a SQL query to find customers with purchases > $1000"
```

### Python API

```python
from data_agent import DataAnalyzer

# Initialize agent
analyzer = DataAnalyzer(model="llama3.2")

# Load data
df = analyzer.load_data("sales_data.csv")

# Ask questions
result = analyzer.query("What are the top selling products?")
print(result.answer)
print(result.visualization)

# Generate report
report = analyzer.generate_report(
    df=df,
    title="Q1 Sales Analysis",
    include_visuals=True
)
report.save("q1_report.pdf")
```

### SQL Query Generation

```python
from data_agent import SQLGenerator

generator = SQLGenerator()

# Convert natural language to SQL
query = generator.generate_sql(
    question="Find all customers who made purchases over $1000 in the last month",
    schema=database_schema
)

# Execute query
results = generator.execute(query, connection)
```

## Features in Detail

### Automated Insights

The agent automatically detects:
- Missing values and data quality issues
- Statistical distributions
- Correlations between variables
- Temporal trends and seasonality
- Outliers and anomalies

### Visualization Types

- Line charts (time series)
- Bar charts (categorical comparisons)
- Scatter plots (correlations)
- Heatmaps (correlation matrices)
- Box plots (distributions)
- Histograms (frequency distributions)

### Report Generation

Generated reports include:
- Executive summary
- Key findings and insights
- Data quality assessment
- Statistical analysis
- Visualizations
- Recommendations

## Configuration

Edit `.env` file:

```env
# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2

# Database Configuration
DB_TYPE=postgresql
DB_HOST=localhost
DB_PORT=5432
DB_NAME=analytics

# Analysis Settings
MAX_FILE_SIZE_MB=100
AUTO_DETECT_ANOMALIES=true
CONFIDENCE_THRESHOLD=0.95
```

## Docker Deployment

```bash
# Build and run
docker-compose up -d

# Access the app
# http://localhost:8501

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## Advanced Usage

### Custom Analysis Pipeline

```python
from data_agent import AnalysisPipeline

pipeline = AnalysisPipeline([
    'load_data',
    'clean_data',
    'detect_outliers',
    'generate_features',
    'analyze_trends',
    'create_visualizations',
    'generate_report'
])

result = pipeline.run('sales_data.csv')
```

### Time Series Forecasting

```python
from data_agent import TimeSeriesAnalyzer

ts_analyzer = TimeSeriesAnalyzer()

# Forecast next 30 days
forecast = ts_analyzer.forecast(
    data=df['sales'],
    periods=30,
    confidence_interval=0.95
)

ts_analyzer.plot_forecast(forecast)
```

## Performance

- **Small datasets (< 1MB):** < 2 seconds
- **Medium datasets (1-50MB):** 5-15 seconds
- **Large datasets (50-100MB):** 30-60 seconds
- **SQL query generation:** < 3 seconds

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Test specific module
pytest tests/test_analyzer.py
```

## Roadmap

- [ ] Support for more data formats (JSON, Parquet, XML)
- [ ] Real-time data streaming analysis
- [ ] Advanced ML model training interface
- [ ] Multi-language support
- [ ] Collaborative features (sharing reports)
- [ ] Integration with BI tools (Tableau, Power BI)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) file.

## Support

- Documentation: [useagenticai.in](https://useagenticai.in)
- Issues: [GitHub Issues](https://github.com/AgenticAI-Ind/data-analysis-agent/issues)
- Email: info@useagenticai.in

---

Built with ❤️ by the AgenticAI team
