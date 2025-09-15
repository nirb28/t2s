#!/usr/bin/env python3
"""
Vanna AI application with NVIDIA LLM and FAISS vector store for Chinook database.
This module sets up the Vanna AI framework to generate SQL queries from natural language
using NVIDIA's hosted LLM and FAISS for vector storage of training data.
"""

import os
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from dotenv import load_dotenv

import vanna
from vanna.remote import VannaDefault
from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore

# Load environment variables
load_dotenv()

class NVIDIAVannaAI(ChromaDB_VectorStore, OpenAI_Chat):
    """
    Custom Vanna AI implementation using NVIDIA's hosted LLM and ChromaDB (FAISS-backed) vector store.
    This class combines NVIDIA's language model capabilities with efficient vector storage
    for SQL query generation from natural language.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the NVIDIA Vanna AI instance.
        
        Args:
            config: Configuration dictionary with API keys and settings
        """
        # Set up NVIDIA API configuration
        nvidia_api_key = config.get('nvidia_api_key') if config else os.getenv('NVIDIA_API_KEY')
        if not nvidia_api_key:
            raise ValueError("NVIDIA API key is required. Set NVIDIA_API_KEY environment variable or pass in config.")
        
        # Initialize ChromaDB vector store
        ChromaDB_VectorStore.__init__(self, config={
            'path': './vanna_vectordb',
            'model': 'all-MiniLM-L6-v2'  # Sentence transformer model for embeddings
        })
        
        # Initialize OpenAI chat (compatible with NVIDIA API)
        # Use environment variable override for OpenAI base URL compatibility
        import os
        original_base_url = os.environ.get('OPENAI_BASE_URL')
        os.environ['OPENAI_BASE_URL'] = config.get('api_base', 'https://integrate.api.nvidia.com/v1') if config else 'https://integrate.api.nvidia.com/v1'
        
        try:
            # Get model from config, then env var, then default
            model = config.get('model') if config else None
            if not model:
                model = os.getenv('NVIDIA_MODEL', 'meta/llama-3.3-70b-instruct')
            
            # Get temperature from config, then env var, then default  
            temperature = config.get('temperature') if config else None
            if temperature is None:
                temperature = float(os.getenv('NVIDIA_TEMPERATURE', '0.1'))
            print(f"using model: {model}" )                
            OpenAI_Chat.__init__(self, config={
                'api_key': nvidia_api_key,
                'model': model,
                'temperature': temperature
            })
        finally:
            # Restore original base URL
            if original_base_url:
                os.environ['OPENAI_BASE_URL'] = original_base_url
            else:
                os.environ.pop('OPENAI_BASE_URL', None)
        
        # Database connection
        self.db_path = None
        self.connection = None

class ChinookVannaApp:
    """
    Main application class for the Chinook database Vanna AI implementation.
    Handles database setup, model training, and query processing.
    """
    
    def __init__(self, nvidia_api_key: str, db_path: str = "chinook.db"):
        """
        Initialize the Chinook Vanna application.
        
        Args:
            nvidia_api_key: NVIDIA API key for LLM access
            db_path: Path to the Chinook SQLite database
        """
        self.db_path = Path(db_path)
        self.nvidia_api_key = nvidia_api_key
        
        # Initialize Vanna AI with NVIDIA configuration
        self.vn = NVIDIAVannaAI(config={
            'nvidia_api_key': nvidia_api_key,
            'model': 'meta/llama-3.3-70b-instruct',
            'api_base': 'https://integrate.api.nvidia.com/v1',
            'temperature': 0.1
        })
        
        # Connect to database
        self.connect_to_database()
        
        # Training status
        self.is_trained = False

    def connect_to_database(self):
        """Connect to the Chinook SQLite database."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found at {self.db_path}. Run setup_database.py first.")
        
        try:
            self.vn.connect_to_sqlite(str(self.db_path))
            print(f"✅ Connected to database: {self.db_path}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to database: {e}")

    def get_database_schema(self) -> Dict[str, Any]:
        """Get the database schema information."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get table information
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [table[0] for table in cursor.fetchall()]
            
            schema = {}
            for table in tables:
                # Get column information
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                
                # Get sample data
                cursor.execute(f"SELECT * FROM {table} LIMIT 3")
                sample_data = cursor.fetchall()
                
                schema[table] = {
                    'columns': [(col[1], col[2]) for col in columns],  # (name, type)
                    'sample_data': sample_data
                }
            
            conn.close()
            return schema
            
        except Exception as e:
            print(f"Error getting schema: {e}")
            return {}

    def train_model(self, force_retrain: bool = False):
        """
        Train the Vanna model with Chinook database schema and sample questions.
        
        Args:
            force_retrain: Whether to force retraining even if already trained
        """
        if self.is_trained and not force_retrain:
            print("📚 Model already trained. Use force_retrain=True to retrain.")
            return
        
        print("🎓 Training Vanna model with Chinook database...")
        
        # Train on database schema
        print("📋 Training on database schema...")
        df_ddl = self.vn.run_sql("SELECT sql FROM sqlite_master WHERE type='table'")
        
        for ddl in df_ddl['sql'].dropna():
            self.vn.train(ddl=ddl)
        
        # Train on sample questions and SQL pairs
        print("❓ Training on sample questions...")
        training_data = self.get_training_data()
        
        for item in training_data:
            if 'question' in item and 'sql' in item:
                self.vn.train(question=item['question'], sql=item['sql'])
        
        print("📊 Training on sample data...")
        # Train on some sample data for context
        sample_queries = [
            "SELECT * FROM artists LIMIT 5",
            "SELECT * FROM albums LIMIT 5", 
            "SELECT * FROM tracks LIMIT 5",
            "SELECT * FROM customers LIMIT 5"
        ]
        
        for query in sample_queries:
            try:
                df = self.vn.run_sql(query)
                self.vn.train(df=df)
            except Exception as e:
                print(f"Warning: Could not train on query {query}: {e}")
        
        self.is_trained = True
        print("✅ Model training completed!")

    def get_training_data(self) -> List[Dict[str, str]]:
        """Get sample training questions and SQL pairs for Chinook database."""
        return [
            {
                "question": "How many artists are in the database?",
                "sql": "SELECT COUNT(*) FROM artists"
            },
            {
                "question": "What are the names of all artists?",
                "sql": "SELECT Name FROM artists ORDER BY Name"
            },
            {
                "question": "How many albums does each artist have?",
                "sql": "SELECT ar.Name, COUNT(al.AlbumId) as AlbumCount FROM artists ar LEFT JOIN albums al ON ar.ArtistId = al.ArtistId GROUP BY ar.ArtistId, ar.Name ORDER BY AlbumCount DESC"
            },
            {
                "question": "What are the top 10 most expensive tracks?",
                "sql": "SELECT Name, UnitPrice FROM tracks ORDER BY UnitPrice DESC LIMIT 10"
            },
            {
                "question": "Which customers have spent the most money?",
                "sql": "SELECT c.CustomerId, c.FirstName, c.LastName, SUM(i.Total) as TotalSpent FROM customers c JOIN invoices i ON c.CustomerId = i.CustomerId GROUP BY c.CustomerId ORDER BY TotalSpent DESC"
            },
            {
                "question": "What are the different genres available?",
                "sql": "SELECT Name FROM genres ORDER BY Name"
            },
            {
                "question": "How many tracks are there in each genre?",
                "sql": "SELECT g.Name, COUNT(t.TrackId) as TrackCount FROM genres g LEFT JOIN tracks t ON g.GenreId = t.GenreId GROUP BY g.GenreId, g.Name ORDER BY TrackCount DESC"
            },
            {
                "question": "What are the top selling tracks?",
                "sql": "SELECT t.Name, COUNT(il.InvoiceLineId) as TimesPurchased FROM tracks t JOIN invoice_items il ON t.TrackId = il.TrackId GROUP BY t.TrackId ORDER BY TimesPurchased DESC LIMIT 10"
            },
            {
                "question": "Which employees have made the most sales?",
                "sql": "SELECT e.FirstName, e.LastName, COUNT(i.InvoiceId) as SalesCount, SUM(i.Total) as TotalSales FROM employees e JOIN customers c ON e.EmployeeId = c.SupportRepId JOIN invoices i ON c.CustomerId = i.CustomerId GROUP BY e.EmployeeId ORDER BY TotalSales DESC"
            },
            {
                "question": "What is the average track length by genre?",
                "sql": "SELECT g.Name, AVG(t.Milliseconds) as AvgLength FROM genres g JOIN tracks t ON g.GenreId = t.GenreId GROUP BY g.GenreId, g.Name ORDER BY AvgLength DESC"
            }
        ]

    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Ask a natural language question and get SQL query + results.
        
        Args:
            question: Natural language question
            
        Returns:
            Dictionary with question, SQL query, results, and metadata
        """
        if not self.is_trained:
            self.train_model()
        
        try:
            # Generate SQL from natural language
            sql_query = self.vn.generate_sql(question)
            
            # Execute the query
            df_result = self.vn.run_sql(sql_query)
            
            # Generate simple explanation (since generate_explanation doesn't exist)
            explanation = f"Generated SQL query to answer: '{question}'"
            
            return {
                'question': question,
                'sql': sql_query,
                'results': df_result,
                'explanation': explanation,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'question': question,
                'sql': None,
                'results': None,
                'explanation': None,
                'success': False,
                'error': str(e)
            }

    def get_similar_questions(self, question: str, n: int = 5) -> List[str]:
        """
        Get similar questions from the training data.
        
        Args:
            question: Input question
            n: Number of similar questions to return
            
        Returns:
            List of similar questions
        """
        try:
            similar = self.vn.get_similar_question_sql(question)
            return [item['question'] for item in similar[:n] if 'question' in item]
        except Exception as e:
            print(f"Error getting similar questions: {e}")
            return []

def main():
    """Main function for testing the Vanna application."""
    # Check for required environment variables
    nvidia_api_key = os.getenv('NVIDIA_API_KEY')
    if not nvidia_api_key:
        print("❌ NVIDIA_API_KEY environment variable not set.")
        print("Please set it in your .env file or environment.")
        return
    
    try:
        # Initialize the app
        app = ChinookVannaApp(nvidia_api_key)
        
        # Train the model
        app.train_model()
        
        # Test with a sample question
        test_question = "How many artists are in the database?"
        print(f"\n🤖 Testing with question: {test_question}")
        
        result = app.ask_question(test_question)
        
        if result['success']:
            print(f"✅ SQL: {result['sql']}")
            print(f"📊 Results:\n{result['results']}")
            if result['explanation']:
                print(f"💡 Explanation: {result['explanation']}")
        else:
            print(f"❌ Error: {result['error']}")
            
    except Exception as e:
        print(f"❌ Application error: {e}")

if __name__ == "__main__":
    main()
