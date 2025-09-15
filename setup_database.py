#!/usr/bin/env python3
"""
Script to download and setup the Chinook SQLite database for the Vanna AI project.
The Chinook database represents a digital media store with tables for artists, albums, 
media tracks, invoices, and customers.
"""

import os
import requests
import sqlite3
from pathlib import Path

def download_chinook_database():
    """Download the Chinook SQLite database if it doesn't exist."""
    db_path = Path("chinook.db")
    
    if db_path.exists():
        print(f"✅ Chinook database already exists at {db_path}")
        return str(db_path)
    
    print("📥 Downloading Chinook SQLite database...")
    
    # URL for the Chinook SQLite database
    url = "https://www.sqlitetutorial.net/wp-content/uploads/2018/03/chinook.zip"
    
    try:
        # Download the zip file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Save as zip file temporarily
        zip_path = Path("chinook.zip")
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("📦 Extracting database...")
        
        # Extract the zip file
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('.')
        
        # Clean up zip file
        zip_path.unlink()
        
        print(f"✅ Chinook database downloaded and extracted to {db_path}")
        return str(db_path)
        
    except Exception as e:
        print(f"❌ Error downloading database: {e}")
        print("📋 Manual setup instructions:")
        print("1. Go to https://www.sqlitetutorial.net/sqlite-sample-database/")
        print("2. Download the Chinook database")
        print("3. Extract chinook.db to this directory")
        return None

def verify_database(db_path):
    """Verify the database is properly set up and show basic info."""
    if not db_path or not Path(db_path).exists():
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print(f"\n📊 Database verification successful!")
        print(f"Found {len(tables)} tables:")
        
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"  • {table_name}: {count} records")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database verification failed: {e}")
        return False

def main():
    """Main function to set up the database."""
    print("🎵 Chinook Database Setup for Vanna AI")
    print("=" * 40)
    
    db_path = download_chinook_database()
    
    if verify_database(db_path):
        print("\n🎉 Setup complete! You can now run the Vanna application.")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
