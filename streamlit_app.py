#!/usr/bin/env python3
"""
Streamlit UI for Vanna AI Chinook Database Query Interface.
This provides a web-based chat interface for querying the Chinook database
using natural language powered by NVIDIA LLM and FAISS vector store.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os
from dotenv import load_dotenv
import time

from vanna_app import ChinookVannaApp

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="🎵 Chinook AI Query Assistant",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #4ECDC4;
        background-color: #f8f9fa;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    
    .assistant-message {
        background-color: #f3e5f5;
        border-left-color: #9c27b0;
    }
    
    .sql-code {
        background-color: #263238;
        color: #ffffff;
        padding: 1rem;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
        margin: 0.5rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'vanna_app' not in st.session_state:
        st.session_state.vanna_app = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'is_trained' not in st.session_state:
        st.session_state.is_trained = False
    if 'nvidia_api_key' not in st.session_state:
        st.session_state.nvidia_api_key = os.getenv('NVIDIA_API_KEY', '')

def setup_sidebar():
    """Setup the sidebar with configuration and information."""
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key input
        nvidia_api_key = st.text_input(
            "NVIDIA API Key",
            value=st.session_state.nvidia_api_key,
            type="password",
            help="Enter your NVIDIA API key for LLM access"
        )
        st.session_state.nvidia_api_key = nvidia_api_key
        
        # Database status
        st.header("📊 Database Status")
        db_path = Path("chinook.db")
        if db_path.exists():
            st.success("✅ Chinook database found")
            
            # Show database stats
            try:
                import sqlite3
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                st.info(f"📋 {len(tables)} tables available")
                
                # Show table details
                with st.expander("View Tables"):
                    for table in tables:
                        cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
                        count = cursor.fetchone()[0]
                        st.write(f"• **{table[0]}**: {count:,} records")
                
                conn.close()
            except Exception as e:
                st.error(f"Error reading database: {e}")
        else:
            st.error("❌ Database not found")
            st.info("Run `python setup_database.py` to download the Chinook database")
        
        # Model status
        st.header("🤖 Model Status")
        if st.session_state.vanna_app and st.session_state.is_trained:
            st.success("✅ Model trained and ready")
        else:
            st.warning("⏳ Model not trained yet")
        
        # Sample questions
        st.header("💡 Sample Questions")
        sample_questions = [
            "How many artists are in the database?",
            "What are the top 10 selling tracks?",
            "Which customers have spent the most money?",
            "What genres are available?",
            "Show me the longest tracks",
            "Which employees made the most sales?"
        ]
        
        for question in sample_questions:
            if st.button(question, key=f"sample_{hash(question)}"):
                st.session_state.question_input = question
                st.rerun()

def initialize_vanna_app():
    """Initialize the Vanna application."""
    if not st.session_state.nvidia_api_key:
        st.error("🔑 Please enter your NVIDIA API key in the sidebar")
        return False
    
    if not Path("chinook.db").exists():
        st.error("📊 Chinook database not found. Please run `python setup_database.py` first.")
        return False
    
    try:
        with st.spinner("🚀 Initializing Vanna AI..."):
            st.session_state.vanna_app = ChinookVannaApp(st.session_state.nvidia_api_key)
        return True
    except Exception as e:
        st.error(f"❌ Failed to initialize Vanna AI: {e}")
        return False

def train_model():
    """Train the Vanna model."""
    if not st.session_state.vanna_app:
        return False
    
    try:
        with st.spinner("🎓 Training the model... This may take a few minutes."):
            st.session_state.vanna_app.train_model()
            st.session_state.is_trained = True
        st.success("✅ Model training completed!")
        return True
    except Exception as e:
        st.error(f"❌ Training failed: {e}")
        return False

def display_chat_history():
    """Display the chat history."""
    for i, chat in enumerate(st.session_state.chat_history):
        # User question
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>🙋 You:</strong> {chat['question']}
        </div>
        """, unsafe_allow_html=True)
        
        # Assistant response
        if chat['success']:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🤖 Assistant:</strong><br>
                <div class="sql-code">
                    <strong>SQL Query:</strong><br>
                    {chat['sql']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display results
            if chat['results'] is not None and not chat['results'].empty:
                st.subheader("📊 Query Results")
                st.dataframe(chat['results'], width='stretch')
                
                # Auto-generate charts for numeric data
                if len(chat['results'].columns) >= 2:
                    try:
                        numeric_cols = chat['results'].select_dtypes(include=['number']).columns
                        if len(numeric_cols) >= 1:
                            create_auto_chart(chat['results'], chat['question'])
                    except Exception as e:
                        st.warning(f"Could not generate chart: {e}")
            
            # Display explanation if available
            if chat.get('explanation'):
                st.info(f"💡 **Explanation:** {chat['explanation']}")
        else:
            st.markdown(f"""
            <div class="error-box">
                <strong>❌ Error:</strong> {chat['error']}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")

def create_auto_chart(df, question):
    """Automatically create charts based on the data."""
    try:
        if len(df) == 0:
            return
        
        # Determine chart type based on data and question
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
            # Bar chart for categorical vs numeric
            if len(df) <= 20:  # Only for reasonable number of bars
                fig = px.bar(
                    df.head(10), 
                    x=categorical_cols[0], 
                    y=numeric_cols[0],
                    title=f"Chart: {question}"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif len(numeric_cols) >= 2:
            # Scatter plot for two numeric columns
            fig = px.scatter(
                df.head(50), 
                x=numeric_cols[0], 
                y=numeric_cols[1],
                title=f"Chart: {question}"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif len(numeric_cols) == 1 and len(df) == 1:
            # Single value - gauge chart
            value = df[numeric_cols[0]].iloc[0]
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = value,
                title = {'text': question},
                gauge = {'axis': {'range': [None, value * 1.5]}}
            ))
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        # Silently fail chart generation
        pass

def process_question(question):
    """Process a user question and get the response."""
    if not st.session_state.vanna_app:
        if not initialize_vanna_app():
            return
    
    if not st.session_state.is_trained:
        if not train_model():
            return
    
    with st.spinner("🤔 Thinking... Generating SQL query..."):
        result = st.session_state.vanna_app.ask_question(question)
    
    # Add to chat history
    st.session_state.chat_history.append(result)

def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🎵 Chinook AI Query Assistant</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        Ask questions about the Chinook music database in natural language!<br>
        Powered by <strong>NVIDIA LLM</strong> and <strong>FAISS Vector Store</strong>
    </div>
    """, unsafe_allow_html=True)
    
    # Setup sidebar
    setup_sidebar()
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Chat interface
        st.header("💬 Chat Interface")
        
        # Question input - use session state value if set from sample questions
        question = st.text_input(
            "Ask a question about the music database:",
            placeholder="e.g., How many artists are in the database?",
            key="question_input",
            value=st.session_state.get("question_input", "")
        )
        
        # Process button
        if st.button("🚀 Ask Question", type="primary") and question.strip():
            process_question(question.strip())
            st.rerun()
        
        # Display chat history
        if st.session_state.chat_history:
            st.header("📝 Conversation History")
            display_chat_history()
            
            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.chat_history = []
                st.rerun()
    
    with col2:
        # Quick actions
        st.header("⚡ Quick Actions")
        
        if st.button("🔄 Retrain Model"):
            if st.session_state.vanna_app:
                train_model()
                st.rerun()
        
        if st.button("📊 Show Database Schema"):
            if st.session_state.vanna_app:
                schema = st.session_state.vanna_app.get_database_schema()
                st.json(schema)
        
        # Export results
        if st.session_state.chat_history:
            st.header("📥 Export")
            if st.button("💾 Export Chat History"):
                import json
                export_data = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'chat_history': st.session_state.chat_history
                }
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(export_data, indent=2, default=str),
                    file_name=f"chinook_chat_history_{int(time.time())}.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()
