#!/usr/bin/env python3
"""
Test script to validate remote embedding integration for Vanna AI.
This script tests the RemoteEmbeddingVectorStore class and ensures it can
communicate with the dsp_ai_rag2 model server for embeddings.
"""

import os
import sys
import requests
from pathlib import Path
from dotenv import load_dotenv

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
load_dotenv()

def test_embedding_server_connection():
    """Test connection to the remote embedding server."""
    print("🔌 Testing embedding server connection...")
    
    embedding_server_url = os.getenv('EMBEDDING_SERVER_URL', 'http://localhost:8080')
    
    try:
        # Test health endpoint
        health_url = f"{embedding_server_url}/health"
        response = requests.get(health_url, timeout=10)
        response.raise_for_status()
        
        health_data = response.json()
        print(f"✅ Embedding server is healthy: {health_data}")
        
        # Test list models endpoint
        models_url = f"{embedding_server_url}/list"
        response = requests.get(models_url, timeout=10)
        response.raise_for_status()
        
        models_data = response.json()
        print(f"✅ Available models: {models_data.get('available_models', [])}")
        print(f"✅ Loaded models: {models_data.get('loaded_models', [])}")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error connecting to embedding server at {embedding_server_url}: {e}")
        print("   Make sure the dsp_ai_rag2 model server is running on port 8080")
        return False
    except Exception as e:
        print(f"❌ Error testing embedding server: {e}")
        return False

def test_remote_embedding_generation():
    """Test remote embedding generation."""
    print("\n🧮 Testing remote embedding generation...")
    
    embedding_server_url = os.getenv('EMBEDDING_SERVER_URL', 'http://localhost:8080')
    model_name = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    
    try:
        # Test embeddings endpoint
        embeddings_url = f"{embedding_server_url}/embeddings"
        test_texts = [
            "What is the total sales for each customer?",
            "SELECT customer_id, SUM(total) FROM invoices GROUP BY customer_id"
        ]
        
        payload = {
            "texts": test_texts,
            "model_name": model_name
        }
        
        response = requests.post(embeddings_url, json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        embeddings = data.get('embeddings', [])
        dimensions = data.get('dimensions', 0)
        
        print(f"✅ Successfully generated embeddings")
        print(f"   Number of embeddings: {len(embeddings)}")
        print(f"   Embedding dimensions: {dimensions}")
        print(f"   Model used: {data.get('model', 'unknown')}")
        
        # Validate embedding structure
        if len(embeddings) != len(test_texts):
            print(f"❌ Expected {len(test_texts)} embeddings, got {len(embeddings)}")
            return False
        
        for i, embedding in enumerate(embeddings):
            if not isinstance(embedding, list) or len(embedding) != dimensions:
                print(f"❌ Invalid embedding structure for text {i}")
                return False
        
        print("✅ All embeddings have correct structure")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error generating embeddings: {e}")
        return False
    except Exception as e:
        print(f"❌ Error in embedding generation test: {e}")
        return False

def test_remote_embedding_vector_store():
    """Test the RemoteEmbeddingVectorStore class."""
    print("\n📚 Testing RemoteEmbeddingVectorStore class...")
    
    try:
        from vanna_app import RemoteEmbeddingVectorStore
        
        # Initialize remote embedding vector store
        embedding_server_url = os.getenv('EMBEDDING_SERVER_URL', 'http://localhost:8080')
        embedding_model = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        
        vector_store = RemoteEmbeddingVectorStore(config={
            'embedding_server_url': embedding_server_url,
            'model_name': embedding_model,
            'storage_path': './test_remote_embeddings'
        })
        
        print("✅ RemoteEmbeddingVectorStore initialized successfully")
        
        # Test adding DDL
        test_ddl = """
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL
        );
        """
        
        ddl_id = vector_store.add_ddl(test_ddl)
        print(f"✅ Added DDL with ID: {ddl_id}")
        
        # Test adding question-SQL pair
        test_question = "How many customers are there?"
        test_sql = "SELECT COUNT(*) FROM customers;"
        
        qs_id = vector_store.add_question_sql(test_question, test_sql)
        print(f"✅ Added question-SQL pair with ID: {qs_id}")
        
        # Test adding documentation
        test_doc = "The customers table contains customer information including names and email addresses."
        
        doc_id = vector_store.add_documentation(test_doc)
        print(f"✅ Added documentation with ID: {doc_id}")
        
        # Test retrieval
        related_ddl = vector_store.get_related_ddl("Show me customer information")
        print(f"✅ Found {len(related_ddl)} related DDL statements")
        
        similar_questions = vector_store.get_similar_question_sql("Count all customers")
        print(f"✅ Found {len(similar_questions)} similar question-SQL pairs")
        
        related_docs = vector_store.get_related_documentation("customer data")
        print(f"✅ Found {len(related_docs)} related documentation items")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing RemoteEmbeddingVectorStore: {e}")
        return False

def test_nvidia_vanna_ai():
    """Test the updated NVIDIAVannaAI class."""
    print("\n🤖 Testing NVIDIAVannaAI with remote embeddings...")
    
    # Check for required environment variables
    nvidia_api_key = os.getenv('NVIDIA_API_KEY')
    if not nvidia_api_key or nvidia_api_key == 'your_nvidia_api_key_here':
        print("⚠️ NVIDIA_API_KEY not set or using default value")
        print("   Skipping NVIDIAVannaAI test (requires valid API key)")
        return True
    
    try:
        from vanna_app import NVIDIAVannaAI
        
        # Initialize with remote embeddings
        embedding_server_url = os.getenv('EMBEDDING_SERVER_URL', 'http://localhost:8080')
        embedding_model = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        
        vanna_ai = NVIDIAVannaAI(config={
            'nvidia_api_key': nvidia_api_key,
            'embedding_server_url': embedding_server_url,
            'embedding_model': embedding_model,
            'model': 'meta/llama-3.3-70b-instruct',
            'temperature': 0.1
        })
        
        print("✅ NVIDIAVannaAI initialized successfully with remote embeddings")
        
        # Test adding training data
        test_ddl = """
        CREATE TABLE test_table (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        );
        """
        
        ddl_id = vanna_ai.add_ddl(test_ddl)
        print(f"✅ Added test DDL via NVIDIAVannaAI: {ddl_id}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing NVIDIAVannaAI: {e}")
        return False

def main():
    """Run all integration tests."""
    print("🧪 Remote Embedding Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("Embedding Server Connection", test_embedding_server_connection),
        ("Remote Embedding Generation", test_remote_embedding_generation),
        ("RemoteEmbeddingVectorStore Class", test_remote_embedding_vector_store),
        ("NVIDIAVannaAI Integration", test_nvidia_vanna_ai)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 Test Results Summary:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Remote embedding integration is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    main()
