# 🎵 Chinook AI Query Assistant

A natural language to SQL query interface for the Chinook music database using **Vanna AI**, **NVIDIA LLM**, and **FAISS vector store**. Ask questions about artists, albums, tracks, customers, and sales in plain English!

![Vanna AI](https://img.shields.io/badge/Vanna-AI-blue)
![NVIDIA](https://img.shields.io/badge/NVIDIA-LLM-green)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)

## 🚀 Features

- **Natural Language Queries**: Ask questions in plain English
- **NVIDIA LLM Integration**: Powered by Meta Llama 3.1 405B Instruct
- **FAISS Vector Store**: Efficient similarity search for training data
- **Interactive Web UI**: Beautiful Streamlit interface with chat history
- **Auto-Generated Charts**: Visualizations for query results
- **Chinook Database**: Complete music store database with 11 tables
- **Pre-trained Examples**: 10+ sample questions to get you started

## 📋 Prerequisites

- Python 3.8+
- NVIDIA API key (get from [build.nvidia.com](https://build.nvidia.com/))
- Internet connection (for database download and API calls)

## ⚡ Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd t2s
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your NVIDIA API key
NVIDIA_API_KEY=your_nvidia_api_key_here
```

### 4. Download Database

```bash
python setup_database.py
```

### 5. Run the Application

```bash
streamlit run streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

## 🛠️ Detailed Setup

### Getting NVIDIA API Key

1. Go to [NVIDIA Build](https://build.nvidia.com/)
2. Sign up or log in to your NVIDIA account
3. Navigate to the API section
4. Generate a new API key
5. Copy the key to your `.env` file

### Database Setup

The Chinook database represents a digital media store with the following tables:

- **artists** - Music artists (275 records)
- **albums** - Music albums (347 records)  
- **tracks** - Individual tracks (3,503 records)
- **customers** - Store customers (59 records)
- **employees** - Store employees (8 records)
- **invoices** - Customer purchases (412 records)
- **invoice_items** - Purchase line items (2,240 records)
- **genres** - Music genres (25 records)
- **media_types** - Track formats (5 records)
- **playlists** - Customer playlists (18 records)
- **playlist_track** - Playlist contents (8,715 records)

## 💬 Usage Examples

### Sample Questions

Try asking these questions in the chat interface:

**Basic Queries:**
- "How many artists are in the database?"
- "What are the different genres available?"
- "Show me all customers from USA"

**Analytics Questions:**
- "Which customers have spent the most money?"
- "What are the top 10 selling tracks?"
- "Which employees made the most sales?"
- "What is the average track length by genre?"

**Complex Queries:**
- "Show me artists with more than 10 albums"
- "Which genres generate the most revenue?"
- "What are the most popular playlists?"

### Chat Interface Features

- **Real-time SQL Generation**: See the generated SQL query for each question
- **Interactive Results**: Browse query results in a searchable table
- **Auto-generated Charts**: Automatic visualizations for numeric data
- **Query History**: Review all previous questions and results
- **Export Options**: Download chat history as JSON
- **Sample Questions**: Quick-start buttons for common queries

## 🏗️ Architecture

### Core Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI   │───▶│   Vanna App      │───▶│ Chinook SQLite  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  NVIDIA LLM API  │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ FAISS Vector DB  │
                    └──────────────────┘
```

### Technology Stack

- **Frontend**: Streamlit with custom CSS styling
- **Backend**: Vanna AI framework
- **LLM**: NVIDIA Meta Llama 3.1 405B Instruct
- **Vector Store**: ChromaDB with FAISS backend
- **Database**: SQLite (Chinook sample database)
- **Embeddings**: all-MiniLM-L6-v2 sentence transformer

## 📁 Project Structure

```
t2s/
├── streamlit_app.py       # Main Streamlit application
├── vanna_app.py          # Vanna AI wrapper and business logic
├── setup_database.py     # Database download and setup script
├── requirements.txt      # Python dependencies
├── .env.example         # Environment configuration template
├── README.md           # This file
├── chinook.db          # SQLite database (created by setup)
└── vanna_vectordb/     # FAISS vector store (created at runtime)
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NVIDIA_API_KEY` | NVIDIA API key for LLM access | Required |
| `NVIDIA_MODEL` | Model name | `meta/llama-3.1-405b-instruct` |
| `NVIDIA_API_BASE` | API endpoint | `https://integrate.api.nvidia.com/v1` |
| `NVIDIA_TEMPERATURE` | Generation temperature | `0.1` |

### Model Training

The application automatically trains the Vanna model on:

1. **Database Schema**: All table structures and relationships
2. **Sample Questions**: 10 pre-defined question/SQL pairs
3. **Sample Data**: Representative rows from each table

Training happens automatically on first use and takes 2-3 minutes.

## 🔍 Troubleshooting

### Common Issues

**1. "NVIDIA_API_KEY not found"**
- Ensure your `.env` file contains the API key
- Verify the key is correct by testing at build.nvidia.com

**2. "Database not found"**
- Run `python setup_database.py` to download the Chinook database
- Check that `chinook.db` exists in the project directory

**3. "Model training failed"**
- Check your internet connection
- Verify NVIDIA API key has sufficient credits
- Try restarting the application

**4. "ChromaDB connection error"**
- Delete the `vanna_vectordb` directory and restart
- Check disk space availability

### Debug Mode

For detailed debugging, run:

```bash
# Test the core application
python vanna_app.py

# Check database setup
python setup_database.py
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional sample questions for training
- Support for other databases
- Enhanced chart generation
- Query optimization suggestions
- Multi-language support

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [Vanna AI](https://github.com/vanna-ai/vanna) - SQL generation framework
- [NVIDIA](https://build.nvidia.com/) - LLM API and hosting
- [Chinook Database](https://github.com/lerocha/chinook-database) - Sample music database
- [Streamlit](https://streamlit.io/) - Web application framework

---

**Happy querying! 🎵**
