#!/usr/bin/env python3
"""
Simple test script to verify the backend is working
"""

import requests
import json

def test_backend():
    base_url = "http://localhost:5001"
    
    print("Testing P4P Backend...")
    print("=" * 30)
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check: {data['message']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend. Make sure it's running on port 5001")
        return
    
    # Test list files endpoint
    try:
        response = requests.get(f"{base_url}/list-files")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ List files: {data['count']} files found")
        else:
            print(f"❌ List files failed: {response.status_code}")
    except Exception as e:
        print(f"❌ List files error: {e}")
    
    print("\nBackend is ready! You can now:")
    print("1. Start the frontend: cd frontend && npm start")
    print("2. Or use the dev script: ./start-dev.sh")
    print("3. Access the app at: http://172.23.163.147:3000")

if __name__ == "__main__":
    test_backend()
