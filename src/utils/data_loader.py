"""
Data Loader - Data ingestion and processing utilities
Handles multiple file formats and data sources.
"""

import pandas as pd
from typing import Optional, Dict, Any
from pathlib import Path
import io
from loguru import logger


class DataLoader:
    """
    Universal data loader for multiple formats.

    Supported formats:
    - CSV
    - Excel (XLS, XLSX)
    - JSON
    - Parquet
    - SQL databases
    """

    def __init__(self):
        """Initialize data loader"""
        logger.info("DataLoader initialized")

    def load_file(
        self,
        file_path: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from file (auto-detect format).

        Args:
            file_path: Path to data file
            **kwargs: Format-specific arguments

        Returns:
            Loaded DataFrame
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        logger.info(f"Loading file: {file_path}")

        if suffix == '.csv':
            return self.load_csv(file_path, **kwargs)
        elif suffix in ['.xls', '.xlsx']:
            return self.load_excel(file_path, **kwargs)
        elif suffix == '.json':
            return self.load_json(file_path, **kwargs)
        elif suffix == '.parquet':
            return self.load_parquet(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def load_csv(
        self,
        file_path: str,
        encoding: str = 'utf-8',
        **kwargs
    ) -> pd.DataFrame:
        """
        Load CSV file.

        Args:
            file_path: Path to CSV file
            encoding: File encoding
            **kwargs: pandas read_csv arguments

        Returns:
            DataFrame
        """
        logger.info(f"Loading CSV: {file_path}")

        try:
            df = pd.read_csv(file_path, encoding=encoding, **kwargs)
            logger.success(f"Loaded {len(df)} rows from CSV")
            return df
        except UnicodeDecodeError:
            # Try different encodings
            for enc in ['latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    df = pd.read_csv(file_path, encoding=enc, **kwargs)
                    logger.warning(f"Loaded with {enc} encoding")
                    return df
                except:
                    continue
            raise ValueError("Unable to decode CSV file")

    def load_excel(
        self,
        file_path: str,
        sheet_name: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load Excel file.

        Args:
            file_path: Path to Excel file
            sheet_name: Sheet to load (None for first sheet)
            **kwargs: pandas read_excel arguments

        Returns:
            DataFrame
        """
        logger.info(f"Loading Excel: {file_path}")

        df = pd.read_excel(file_path, sheet_name=sheet_name or 0, **kwargs)
        logger.success(f"Loaded {len(df)} rows from Excel")

        return df

    def load_json(
        self,
        file_path: str,
        orient: str = 'records',
        **kwargs
    ) -> pd.DataFrame:
        """
        Load JSON file.

        Args:
            file_path: Path to JSON file
            orient: JSON orientation
            **kwargs: pandas read_json arguments

        Returns:
            DataFrame
        """
        logger.info(f"Loading JSON: {file_path}")

        df = pd.read_json(file_path, orient=orient, **kwargs)
        logger.success(f"Loaded {len(df)} rows from JSON")

        return df

    def load_parquet(
        self,
        file_path: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load Parquet file.

        Args:
            file_path: Path to Parquet file
            **kwargs: pandas read_parquet arguments

        Returns:
            DataFrame
        """
        logger.info(f"Loading Parquet: {file_path}")

        df = pd.read_parquet(file_path, **kwargs)
        logger.success(f"Loaded {len(df)} rows from Parquet")

        return df

    def load_from_sql(
        self,
        query: str,
        connection_string: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from SQL database.

        Args:
            query: SQL query
            connection_string: Database connection string
            **kwargs: pandas read_sql arguments

        Returns:
            DataFrame
        """
        logger.info(f"Loading from SQL database")

        from sqlalchemy import create_engine

        engine = create_engine(connection_string)
        df = pd.read_sql(query, engine, **kwargs)

        logger.success(f"Loaded {len(df)} rows from SQL")

        return df

    def load_from_bytes(
        self,
        file_bytes: bytes,
        file_type: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from bytes (for uploaded files).

        Args:
            file_bytes: File content as bytes
            file_type: File extension (csv, xlsx, etc.)
            **kwargs: Format-specific arguments

        Returns:
            DataFrame
        """
        logger.info(f"Loading from bytes: {file_type}")

        if file_type == 'csv':
            return pd.read_csv(io.BytesIO(file_bytes), **kwargs)
        elif file_type in ['xls', 'xlsx']:
            return pd.read_excel(io.BytesIO(file_bytes), **kwargs)
        elif file_type == 'json':
            return pd.read_json(io.BytesIO(file_bytes), **kwargs)
        elif file_type == 'parquet':
            return pd.read_parquet(io.BytesIO(file_bytes), **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate loaded DataFrame.

        Args:
            df: DataFrame to validate

        Returns:
            Validation results
        """
        issues = []

        # Check for empty DataFrame
        if len(df) == 0:
            issues.append("DataFrame is empty")

        # Check for duplicate columns
        if len(df.columns) != len(set(df.columns)):
            issues.append("Duplicate column names found")

        # Check for all missing columns
        for col in df.columns:
            if df[col].isnull().all():
                issues.append(f"Column '{col}' is completely empty")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "rows": len(df),
            "columns": len(df.columns),
            "memory_mb": df.memory_usage(deep=True).sum() / 1024**2
        }

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic data cleaning.

        Args:
            df: DataFrame to clean

        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning DataFrame")

        # Remove duplicate rows
        original_len = len(df)
        df = df.drop_duplicates()
        if len(df) < original_len:
            logger.info(f"Removed {original_len - len(df)} duplicate rows")

        # Remove completely empty columns
        df = df.dropna(axis=1, how='all')

        # Strip whitespace from string columns
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip() if df[col].dtype == 'object' else df[col]

        return df


# Example usage
if __name__ == "__main__":
    loader = DataLoader()

    # Example: Load CSV
    # df = loader.load_csv("data.csv")

    # Example: Load Excel
    # df = loader.load_excel("data.xlsx", sheet_name="Sheet1")

    # Example: Validate
    # validation = loader.validate_dataframe(df)
    # print(f"Valid: {validation['valid']}")
    # if validation['issues']:
    #     print(f"Issues: {', '.join(validation['issues'])}")

    print("DataLoader ready for use")
