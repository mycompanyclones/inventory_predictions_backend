#!/usr/bin/env python3
"""
System Startup Script
====================

This script validates the system and starts the Flask application
with proper error handling and logging.
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def print_banner():
    """Print startup banner"""
    print("\n" + "="*60)
    print("🚀 ENHANCED SUPPLY CHAIN BACKEND SYSTEM v2.0")
    print("="*60)
    print("🔧 Features:")
    print("   • Real-time Socket.IO orchestration")
    print("   • Redis-based session management")
    print("   • Comprehensive tool calling")
    print("   • Enhanced error handling")
    print("="*60)
    print(f"🕐 Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

def check_redis():
    """Check if Redis is running"""
    print("\n🔍 Checking Redis connection...")
    try:
        import redis
        client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        client.ping()
        print("✅ Redis connection successful")
        return True
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("💡 To start Redis:")
        print("   - macOS: brew services start redis")
        print("   - Linux: sudo systemctl start redis")
        print("   - Windows: redis-server")
        return False

def check_environment():
    """Check environment setup"""
    print("\n🔍 Checking environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Check required files
    required_files = [
        'supply_chain_data.csv',
        'requirements.txt',
        'agentic_system.py',
        'orchestration_system.py'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} found")
        else:
            print(f"❌ {file} missing")
            return False
    
    # Check environment variables
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  OPENAI_API_KEY not set (some features may not work)")
    else:
        print("✅ OPENAI_API_KEY configured")
    
    return True

def run_system_validation():
    """Run comprehensive system validation"""
    print("\n🔍 Running system validation...")
    
    try:
        # Import and run validator
        from system_validator import main as validate_system
        
        # Capture validation result
        result = validate_system()
        
        if result:
            print("✅ System validation passed")
            return True
        else:
            print("❌ System validation failed")
            return False
            
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

def start_flask_app():
    """Start the Flask application"""
    print("\n🚀 Starting Flask application...")
    
    try:
        # Import the Flask app
        from app import app, socketio
        
        print("🌐 Server starting on http://localhost:5000")
        print("📡 Socket.IO available at ws://localhost:5000/socket.io/")
        print("🔧 Admin endpoints:")
        print("   • GET /health - Health check")
        print("   • GET /admin/sessions - Active sessions")
        print("   • GET /admin/tools - Available tools")
        print("\n💡 Press Ctrl+C to stop the server")
        print("="*60)
        
        # Start the server
        socketio.run(
            app,
            debug=False,
            host='0.0.0.0',
            port=5000,
            use_reloader=False,
            log_output=True
        )
        
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        print("👋 Goodbye!")
        
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        print("🔍 Check the error details above")
        return False

def main():
    """Main startup function"""
    print_banner()
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed")
        sys.exit(1)
    
    # Check Redis
    if not check_redis():
        print("\n❌ Redis check failed")
        print("🔧 Please start Redis server and try again")
        sys.exit(1)
    
    # Run system validation
    print("\n🔍 Would you like to run system validation? (y/N): ", end="")
    try:
        response = input().lower()
        if response in ['y', 'yes']:
            if not run_system_validation():
                print("\n⚠️  System validation failed, but continuing...")
                print("🔧 Some features may not work correctly")
        else:
            print("⏭️  Skipping system validation")
    except KeyboardInterrupt:
        print("\n\n👋 Startup cancelled by user")
        sys.exit(0)
    
    # Start the Flask app
    start_flask_app()

if __name__ == "__main__":
    main() 