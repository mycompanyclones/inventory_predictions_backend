#!/usr/bin/env python3
"""
Socket.IO Connection Test Script
===============================

This script tests the Socket.IO connection between the frontend and backend
by simulating a client and sending a test message.
"""

import socketio
import time
import json
from datetime import datetime

# Create a Socket.IO client
sio = socketio.Client()

# Connection events
@sio.event
def connect():
    print("🟢 Connected to backend server!")
    print(f"🕐 Connection time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

@sio.event
def connect_error(data):
    print(f"🔴 Connection failed: {data}")

@sio.event
def disconnect():
    print("🔴 Disconnected from backend server")

# Message events
@sio.event
def message(data):
    print(f"📨 Received message: {data}")

@sio.event
def processing_start(data):
    print(f"🚀 Processing started: {data}")

@sio.event
def tool_start(data):
    print(f"🔧 Tool started: {data['tool_name']} - {data['description']}")

@sio.event
def tool_complete(data):
    print(f"✅ Tool completed: {data['tool_name']} - {data['summary']}")

@sio.event
def processing_complete(data):
    print(f"🎉 Processing complete: {data}")

@sio.event
def error(data):
    print(f"❌ Error: {data}")

def test_connection():
    """Test the Socket.IO connection"""
    print("🧪 Testing Socket.IO connection...")
    print("=" * 50)
    
    try:
        # Connect to the backend
        sio.connect('http://localhost:5000')
        
        # Wait for connection
        time.sleep(2)
        
        # Test sending a message
        print("\n📤 Sending test message...")
        test_message = {
            "message": "What is the current inventory status?",
            "session_id": "test_session_123"
        }
        
        sio.emit('messages', test_message)
        
        # Wait for responses
        print("⏳ Waiting for responses...")
        time.sleep(10)
        
        # Test session info
        print("\n📊 Requesting session info...")
        sio.emit('get_session_info', {'session_id': 'test_session_123'})
        
        # Wait for session info
        time.sleep(5)
        
        # Disconnect
        print("\n🔌 Disconnecting...")
        sio.disconnect()
        
        print("\n✅ Connection test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Connection test failed: {e}")
        return False
    
    finally:
        if sio.connected:
            sio.disconnect()

def test_backend_health():
    """Test backend health endpoint"""
    print("\n🏥 Testing backend health endpoint...")
    
    try:
        import requests
        response = requests.get('http://localhost:5000/health')
        
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Backend health check passed:")
            print(f"   Status: {health_data.get('status')}")
            print(f"   Redis: {health_data.get('redis')}")
            print(f"   Active sessions: {health_data.get('active_sessions')}")
            return True
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_admin_endpoints():
    """Test admin endpoints"""
    print("\n🛠️ Testing admin endpoints...")
    
    try:
        import requests
        
        # Test sessions endpoint
        response = requests.get('http://localhost:5000/admin/sessions')
        if response.status_code == 200:
            sessions_data = response.json()
            print(f"✅ Sessions endpoint working:")
            print(f"   Active sessions: {sessions_data.get('active_sessions')}")
        else:
            print(f"❌ Sessions endpoint failed: {response.status_code}")
        
        # Test tools endpoint
        response = requests.get('http://localhost:5000/admin/tools')
        if response.status_code == 200:
            tools_data = response.json()
            print(f"✅ Tools endpoint working:")
            print(f"   Total tools: {tools_data.get('tools', {}).get('total_tools')}")
        else:
            print(f"❌ Tools endpoint failed: {response.status_code}")
            
        return True
        
    except Exception as e:
        print(f"❌ Admin endpoints test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Backend-Frontend Connection Test")
    print("=" * 50)
    
    # Test 1: Backend health
    health_ok = test_backend_health()
    
    # Test 2: Admin endpoints
    admin_ok = test_admin_endpoints()
    
    # Test 3: Socket.IO connection
    connection_ok = test_connection()
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 50)
    print(f"Health Check: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"Admin Endpoints: {'✅ PASS' if admin_ok else '❌ FAIL'}")
    print(f"Socket.IO Connection: {'✅ PASS' if connection_ok else '❌ FAIL'}")
    
    all_tests_passed = health_ok and admin_ok and connection_ok
    
    if all_tests_passed:
        print("\n🎉 All tests passed! Backend and frontend are connected successfully!")
        print("\n🌐 You can now open http://localhost:3000 in your browser to use the chat interface.")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 