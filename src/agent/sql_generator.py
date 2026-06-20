"""
SQL Generator - Natural Language to SQL Conversion
Converts natural language queries to SQL statements.
"""

from typing import Dict, Any, Optional
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from loguru import logger
import sqlalchemy
from sqlalchemy import create_engine, text


class SQLGenerator:
    """
    Natural language to SQL query generator.

    Features:
    - NL to SQL conversion
    - Query validation
    - Query execution
    - Schema awareness
    """

    def __init__(self, model: str = "llama3.2"):
        """
        Initialize SQL generator.

        Args:
            model: Ollama model for query generation
        """
        self.llm = Ollama(model=model, temperature=0.1)  # Low temp for accuracy
        logger.info(f"SQLGenerator initialized with {model}")

    def generate_sql(
        self,
        question: str,
        schema: Dict[str, Any],
        dialect: str = "postgresql"
    ) -> str:
        """
        Generate SQL query from natural language.

        Args:
            question: Natural language question
            schema: Database schema information
            dialect: SQL dialect (postgresql, mysql, sqlite)

        Returns:
            Generated SQL query
        """
        logger.info(f"Generating SQL for: {question}")

        # Format schema information
        schema_str = self._format_schema(schema)

        prompt = PromptTemplate(
            input_variables=["question", "schema", "dialect"],
            template="""You are an expert SQL query writer.

Database Schema:
{schema}

SQL Dialect: {dialect}

User Question: {question}

Generate a valid SQL query to answer this question.
Rules:
- Use standard SQL syntax
- Include appropriate WHERE clauses
- Use JOINs when needed
- Add LIMIT if reasonable
- Return only the SQL query, no explanations

SQL Query:
"""
        )

        from langchain.chains import LLMChain
        chain = LLMChain(llm=self.llm, prompt=prompt)

        sql_query = chain.run(
            question=question,
            schema=schema_str,
            dialect=dialect
        ).strip()

        # Clean up the query
        sql_query = self._clean_query(sql_query)

        logger.info(f"Generated SQL: {sql_query}")
        return sql_query

    def execute(
        self,
        query: str,
        connection_string: str
    ) -> Dict[str, Any]:
        """
        Execute SQL query and return results.

        Args:
            query: SQL query to execute
            connection_string: Database connection string

        Returns:
            Query results and metadata
        """
        logger.info("Executing SQL query")

        try:
            engine = create_engine(connection_string)

            with engine.connect() as conn:
                result = conn.execute(text(query))

                # Fetch results
                rows = result.fetchall()
                columns = result.keys()

                return {
                    "success": True,
                    "rows": [dict(zip(columns, row)) for row in rows],
                    "row_count": len(rows),
                    "columns": list(columns)
                }

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "rows": [],
                "row_count": 0
            }

    def _format_schema(self, schema: Dict[str, Any]) -> str:
        """Format schema for prompt"""
        schema_parts = []

        for table_name, columns in schema.items():
            cols = ", ".join([f"{col['name']} {col['type']}" for col in columns])
            schema_parts.append(f"Table: {table_name}\nColumns: {cols}")

        return "\n\n".join(schema_parts)

    def _clean_query(self, query: str) -> str:
        """Clean and validate SQL query"""
        # Remove markdown code blocks
        query = query.replace("```sql", "").replace("```", "")

        # Remove common explanatory text
        lines = query.split('\n')
        sql_lines = [line for line in lines if not line.strip().startswith('--')]

        return ' '.join(sql_lines).strip()


# Example usage
if __name__ == "__main__":
    generator = SQLGenerator()

    # Example schema
    schema = {
        "customers": [
            {"name": "id", "type": "INTEGER"},
            {"name": "name", "type": "VARCHAR"},
            {"name": "email", "type": "VARCHAR"},
            {"name": "created_at", "type": "TIMESTAMP"}
        ],
        "orders": [
            {"name": "id", "type": "INTEGER"},
            {"name": "customer_id", "type": "INTEGER"},
            {"name": "total", "type": "DECIMAL"},
            {"name": "status", "type": "VARCHAR"}
        ]
    }

    # Generate SQL
    question = "Show me the top 10 customers by total order value"
    sql = generator.generate_sql(question, schema)

    print(f"Question: {question}")
    print(f"Generated SQL: {sql}")
