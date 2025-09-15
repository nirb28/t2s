#!/usr/bin/env python3
"""
Test script for the Chinook Vanna AI application.
This script validates the core functionality including database setup,
model initialization, training, and query processing.
"""

import os
import sys
from pathlib import Path
import sqlite3
from dotenv import load_dotenv
import time

def test_environment_setup():
    """Test environment configuration and dependencies."""
    print("🧪 Testing Environment Setup")
    print("-" * 40)
    
    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  WARNING: .env file not found. Copy .env.example to .env and configure.")
        return False
    
    # Load environment variables
    load_dotenv()
    
    # Check NVIDIA API key
    nvidia_api_key = os.getenv('NVIDIA_API_KEY')
    if not nvidia_api_key or nvidia_api_key == 'your_nvidia_api_key_here':
        print("❌ NVIDIA_API_KEY not configured in .env file")
        return False
    
    print("✅ Environment variables configured")
    
    # Test imports
    try:
        import vanna
        import streamlit
        import pandas
        import plotly
        import sqlite3
        print("✅ All required packages imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Run: pip install -r requirements.txt")
        return False

def test_database_setup():
    """Test Chinook database setup and connectivity."""
    print("\n📊 Testing Database Setup")
    print("-" * 40)
    
    db_path = Path("chinook.db")
    if not db_path.exists():
        print("❌ Chinook database not found. Run: python setup_database.py")
        return False
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall()]
        
        expected_tables = ['artists', 'albums', 'tracks', 'customers', 'employees', 
                          'invoices', 'invoice_items', 'genres', 'media_types', 
                          'playlists', 'playlist_track']
        
        missing_tables = set(expected_tables) - set(tables)
        if missing_tables:
            print(f"❌ Missing tables: {missing_tables}")
            return False
        
        # Check data exists
        cursor.execute("SELECT COUNT(*) FROM artists")
        artist_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM tracks") 
        track_count = cursor.fetchone()[0]
        
        if artist_count == 0 or track_count == 0:
            print("❌ Database tables are empty")
            return False
        
        print(f"✅ Database connected successfully")
        print(f"   • {len(tables)} tables found")
        print(f"   • {artist_count} artists, {track_count} tracks")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def test_vanna_initialization():
    """Test Vanna AI application initialization."""
    print("\n🤖 Testing Vanna AI Initialization")
    print("-" * 40)
    
    try:
        from vanna_app import ChinookVannaApp
        
        nvidia_api_key = os.getenv('NVIDIA_API_KEY')
        if not nvidia_api_key:
            print("❌ NVIDIA_API_KEY not available")
            return False
        
        # Initialize application
        print("🚀 Initializing Vanna application...")
        app = ChinookVannaApp(nvidia_api_key)
        
        print("✅ Vanna application initialized successfully")
        return app
        
    except Exception as e:
        print(f"❌ Vanna initialization failed: {e}")
        return None

def test_model_training(app):
    """Test model training process."""
    print("\n🎓 Testing Model Training")
    print("-" * 40)
    
    if not app:
        print("❌ Cannot test training - app not initialized")
        return False
    
    try:
        print("⏳ Training model... (this may take 2-3 minutes)")
        start_time = time.time()
        
        app.train_model()
        
        training_time = time.time() - start_time
        print(f"✅ Model training completed in {training_time:.1f} seconds")
        return True
        
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return False

def test_query_processing(app):
    """Test natural language query processing."""
    print("\n💬 Testing Query Processing")
    print("-" * 40)
    
    if not app:
        print("❌ Cannot test queries - app not initialized")
        return False
    
    test_questions = [
        "How many artists are in the database?",
        "What are the different genres available?",
        "Show me the top 5 tracks by name"
    ]
    
    successful_queries = 0
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n🔍 Test Query {i}: {question}")
        
        try:
            result = app.ask_question(question)
            
            if result['success']:
                print(f"   ✅ SQL: {result['sql'][:100]}...")
                if result['results'] is not None and not result['results'].empty:
                    print(f"   ✅ Results: {len(result['results'])} rows returned")
                    successful_queries += 1
                else:
                    print("   ⚠️  No results returned")
            else:
                print(f"   ❌ Query failed: {result['error']}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    success_rate = (successful_queries / len(test_questions)) * 100
    print(f"\n📊 Query Success Rate: {success_rate:.1f}% ({successful_queries}/{len(test_questions)})")
    
    return successful_queries >= 2  # At least 2/3 queries should work

def test_streamlit_compatibility():
    """Test Streamlit application compatibility."""
    print("\n🌐 Testing Streamlit Compatibility")
    print("-" * 40)
    
    try:
        # Check if streamlit_app.py can be imported
        import streamlit_app
        print("✅ Streamlit application imports successfully")
        
        # Check for main components
        main_function = getattr(streamlit_app, 'main', None)
        if main_function:
            print("✅ Main function found")
        else:
            print("⚠️  Main function not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Streamlit compatibility test failed: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive application test suite."""
    print("🎵 Chinook Vanna AI - Comprehensive Test Suite")
    print("=" * 50)
    
    tests = [
        ("Environment Setup", test_environment_setup),
        ("Database Setup", test_database_setup),
        ("Streamlit Compatibility", test_streamlit_compatibility),
    ]
    
    # Run basic tests first
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    # Only run advanced tests if basics pass
    app = None
    if results["Environment Setup"] and results["Database Setup"]:
        app = test_vanna_initialization()
        results["Vanna Initialization"] = app is not None
        
        if app:
            results["Model Training"] = test_model_training(app)
            if results["Model Training"]:
                results["Query Processing"] = test_query_processing(app)
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    success_rate = (passed / total) * 100
    print(f"\nOverall Success Rate: {success_rate:.1f}% ({passed}/{total})")
    
    if success_rate >= 80:
        print("\n🎉 Application is ready to use!")
        print("Run: streamlit run streamlit_app.py")
    else:
        print("\n⚠️  Some tests failed. Check the output above for issues.")
    
    return success_rate >= 80

def main():
    """Main test function with command line interface."""
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        
        if test_type == "env":
            test_environment_setup()
        elif test_type == "db":
            test_database_setup()
        elif test_type == "vanna":
            app = test_vanna_initialization()
            if app:
                test_model_training(app)
                test_query_processing(app)
        elif test_type == "streamlit":
            test_streamlit_compatibility()
        else:
            print("Usage: python test_application.py [env|db|vanna|streamlit]")
    else:
        # Run comprehensive test
        run_comprehensive_test()

if __name__ == "__main__":
    main()
