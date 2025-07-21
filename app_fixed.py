import os
import traceback
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import redis
import json
import pickle
import uuid
from typing import Dict, Any

# Import the enhanced orchestrator or fall back to basic tools
try:
    from enhanced_orchestrator import EnhancedOrchestrator
    from agentic_system import SessionManager
    ORCHESTRATOR_AVAILABLE = True
except ImportError as e:
    # Fallback to basic system if enhanced orchestrator not available
    print(f"⚠️ Enhanced Orchestrator not available: {e}")
    ORCHESTRATOR_AVAILABLE = False

app = Flask(__name__)

# Configure CORS properly
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000"], supports_credentials=True)

# Configure SocketIO with proper CORS
socketio = SocketIO(
    app, 
    cors_allowed_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    async_mode='threading',
    logger=False,
    engineio_logger=False,
    ping_timeout=120,
    ping_interval=30
)

# Redis configuration for session management
try:
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        decode_responses=True
    )
    redis_client.ping()
    print("✅ Redis connected successfully")
except Exception as redis_error:
    print(f"❌ Redis connection failed: {redis_error}")
    redis_client = None

# Track active WebSocket sessions
active_sessions = {}

# Initialize orchestrator
if ORCHESTRATOR_AVAILABLE:
    orchestrator = EnhancedOrchestrator()
    print("✅ Enhanced Orchestrator initialized")
else:
    print("⚠️ Using fallback tools - Enhanced Orchestrator not available")

@app.route('/')
def index():
    return jsonify({
        "status": "Inventory Predictions Backend Running", 
        "timestamp": datetime.now().isoformat(),
        "redis_available": redis_client is not None,
        "orchestrator_available": ORCHESTRATOR_AVAILABLE
    })

@socketio.on('connect')
def handle_connect():
    try:
        print(f"🔗 Client connected: {request.sid}")
        
        # Create or get existing session
        session_id = request.sid
        
        # Initialize session manager
        session_manager = SessionManager(session_id)
        active_sessions[session_id] = {
            "manager": session_manager,
            "connected_at": datetime.now(),
            "last_activity": datetime.now()
        }
        
        emit('connection_status', {
            "status": "connected", 
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "redis_available": redis_client is not None,
            "orchestrator_available": ORCHESTRATOR_AVAILABLE,
        })
        
    except Exception as e:
        print(f"❌ Error in connect handler: {str(e)}")
        emit('connection_status', {"status": "error", "message": str(e)})

@socketio.on('disconnect')
def handle_disconnect():
    try:
        session_id = request.sid
        print(f"🔌 Client disconnected: {session_id}")
        
        # Clean up session data
        if session_id in active_sessions:
            # Optionally finalize session or save state
            # session_manager = active_sessions[session_id]["manager"]
            # session_manager.finalize_session()
            del active_sessions[session_id]
            
    except Exception as e:
        print(f"❌ Error in disconnect handler: {str(e)}")

@socketio.on('messages')
def handle_messages(data):
    try:
        session_id = request.sid
        user_message = data.get('message', '').strip()
        
        if not user_message:
            emit('response', {'type': 'error', 'content': 'Empty message received'})
            return
        
        print(f"📥 Received message from {session_id}: {user_message}")
        
        # Get or create session manager
        if session_id not in active_sessions:
            session_manager = SessionManager(session_id)
            active_sessions[session_id] = {
                "manager": session_manager,
                "connected_at": datetime.now(),
                "last_activity": datetime.now()
            }
        else:
            session_manager = active_sessions[session_id]["manager"]
            active_sessions[session_id]["last_activity"] = datetime.now()
        
        # Check if session is already processing
        if hasattr(session_manager, 'is_processing') and session_manager.is_processing():
            print(f"⚠️ Session {session_id} is already processing - rejecting new message")
            emit('response', {
                'type': 'error',
                'content': 'Please wait for the current analysis to complete before sending a new message.',
                'streaming_status': 'end',
                'session_ready': False,
                'timestamp': datetime.now().isoformat()
            })
            return
        
        # Mark session as processing
        if hasattr(session_manager, 'set_processing'):
            session_manager.set_processing(True)
        
        session_manager.add_message_to_history({"role": "user", "content": user_message})
        
        # Process the query using orchestrator (emits in real-time)
        if ORCHESTRATOR_AVAILABLE:
            # Use enhanced orchestrator with real-time emissions
            result = process_with_orchestrator(user_message, session_manager, session_id)
            print("result from orchestrator===============================================", result)
        else:
            # Use fallback processing
            result = process_with_fallback(user_message, session_manager)
        
        # Extract response content
        response_content = result.get('response', 'Analysis completed')
        
        # Add assistant response to history
        session_manager.add_message_to_history({"role": "assistant", "content": response_content})
        
        # CRITICAL: Reset processing state so session can handle new queries
        if hasattr(session_manager, '_reset_processing_state'):
            session_manager._reset_processing_state()
        
        # Stream fallback response
        emit('message', {
            'type': 'text',
            'content': response_content,
            'timestamp': datetime.now().isoformat(),
            'sender': 'assistant',
            'streaming_status': 'complete'
        })
        
        # Send completion signal
        emit('response', {
            'type': 'analysis_complete',
            'content': response_content,
            'metadata': {
                'tools_used': result.get('tools_used', []),
                'processing_time': result.get('processing_time', 0),
                'timestamp': datetime.now().isoformat()
            },
            'streaming_status': 'end',
            'session_ready': True,
            'timestamp': datetime.now().isoformat()
        })
        
        print("✅ Message processing completed")
        
    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}"
        print(f"❌ Error in handle_messages: {error_msg}")
        print(f"📊 Traceback: {traceback.format_exc()}")
        
        # CRITICAL: Reset processing state on error
        try:
            if 'session_manager' in locals() and hasattr(session_manager, '_reset_processing_state'):
                session_manager._reset_processing_state()
            
            # Add error to session history
            if 'session_manager' in locals():
                session_manager.add_message_to_history({"role": "assistant", "content": f"Error: {error_msg}"})
            
            # Emit error response
            emit('response', {
                'type': 'error',
                'content': error_msg,
                'streaming_status': 'end',
                'session_ready': True,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as cleanup_error:
            print(f"❌ Error during cleanup: {cleanup_error}")
            emit('response', {
                'type': 'error', 
                'content': 'System error occurred',
                'streaming_status': 'end',
                'session_ready': True,
                'timestamp': datetime.now().isoformat()
            })

def process_with_orchestrator(user_message: str, session_manager: SessionManager, session_id: str) -> Dict[str, Any]:
    """Process query using the enhanced orchestrator"""
    try:
        print(f"🔄 Processing with orchestrator: {user_message}")
        
        # Create callback functions for real-time emissions
        def emit_progress(data):
            """Emit progress updates during processing"""
            emit('response', {
                'type': 'progress',
                'content': data.get('content', ''),
                'metadata': data.get('metadata', {}),
                'streaming_status': 'streaming',
                'timestamp': datetime.now().isoformat()
            })
        
        def emit_tool_start(tool_name, description):
            """Emit when a tool starts execution"""
            emit('response', {
                'type': 'tool_start',
                'content': f"🔧 Executing: {description}",
                'tool_name': tool_name,
                'streaming_status': 'streaming',
                'timestamp': datetime.now().isoformat()
            })
        
        def emit_tool_result(tool_name, result_data):
            """Emit when a tool completes"""
            emit('response', {
                'type': 'tool_result',
                'content': f"✅ {tool_name} completed",
                'result': result_data,
                'streaming_status': 'streaming',
                'timestamp': datetime.now().isoformat()
            })
        
        def emit_analysis(content):
            """Emit analysis content as it's generated"""
            emit('response', {
                'type': 'analysis',
                'content': content,
                'streaming_status': 'streaming',
                'timestamp': datetime.now().isoformat()
            })
        
        # Create callbacks object
        callbacks = {
            'emit_progress': emit_progress,
            'emit_tool_start': emit_tool_start,
            'emit_tool_result': emit_tool_result,
            'emit_analysis': emit_analysis
        }
        
        # Process the query through enhanced orchestrator
        result = orchestrator.process_query_with_streaming(
            user_message,
            session_history=session_manager.get_message_history(),
            session_context=getattr(session_manager, 'get_session_context', lambda: {})(),
            session_id=session_id
        )
        
        return result
        
    except Exception as e:
        return {'response': f'Error in orchestrator processing: {str(e)}', 'tools_used': [], 'processing_time': 0}

def process_with_fallback(user_message: str, session_manager: SessionManager) -> Dict[str, Any]:
    """Fallback processing when orchestrator is not available"""
    return {
        'response': f"""
# ⚠️ Limited Analysis Mode

I'm currently running in limited mode. The query "{user_message}" was received, but advanced analysis tools are not available.

## Current Status
Since the enhanced orchestrator is not available, please ensure:

1. **Check System Requirements**: Enhanced analysis tools may not be properly installed
2. **Verify Dependencies**: Required analysis modules may be missing

## Available Actions
| Feature | Status | Notes |
|---------|--------|--------|
| Basic Response | ✅ Available | Simple text responses work |
| Load Data | ❌ No | Orchestrator not available |
| Process Query | ❌ No | Orchestrator not available |
| Generate Insights | ❌ No | Orchestrator not available |

## Next Steps
1. Check the system configuration
2. Restart the application with proper orchestrator setup
3. Contact administrator if issues persist

*This is a fallback response - full analysis capabilities require the enhanced orchestrator.*
        """,
        'tools_used': [],
        'processing_time': 0,
        'error': 'Enhanced orchestrator not available'
    }

@socketio.on('get_session_info')
def handle_get_session_info(data):
    try:
        session_id = data.get('session_id', request.sid)
        
        if session_id in active_sessions:
            session_manager = active_sessions[session_id]["manager"]
            session_data = session_manager._get_session_data()
            
            session_info = {
                'session_id': session_id,
                'query_count': session_data.get('query_count', 0),
                'message_count': len(session_data.get('message_history', [])),
                'created_at': session_data.get('session_start'),
                'last_activity': session_data.get('last_activity'),
                'connected_at': active_sessions[session_id]["connected_at"].isoformat(),
                'redis_available': redis_client is not None
            }
            
            emit('session_info', session_info)
        else:
            emit('session_info', {'error': 'Session not found'})
            
    except Exception as e:
        print(f"❌ Error getting session info: {str(e)}")
        emit('session_info', {'error': str(e)})

@socketio.on('clear_session')
def handle_clear_session(data):
    try:
        session_id = data.get('session_id', request.sid)
        
        if session_id in active_sessions:
            # Clear the session data
            session_manager = SessionManager(session_id)
            active_sessions[session_id]["manager"] = session_manager
            
            emit('session_cleared', {
                'status': 'success', 
                'session_id': session_id,
                'timestamp': datetime.now().isoformat()
            })
        else:
            emit('session_cleared', {'status': 'error', 'message': 'Session not found'})
            
    except Exception as e:
        print(f"❌ Error clearing session: {str(e)}")
        emit('session_cleared', {'status': 'error', 'message': str(e)})

@app.route('/admin/sessions')
def admin_sessions():
    try:
        sessions_info = {}
        for session_id, session_data in active_sessions.items():
            try:
                session_manager_data = session_data["manager"]._get_session_data()
                sessions_info[session_id] = {
                    "connected_at": session_data["connected_at"].isoformat(),
                    "last_activity": session_data["last_activity"].isoformat(),
                    "query_count": session_manager_data.get('query_count', 0),
                    "message_count": len(session_manager_data.get('message_history', []))
                }
            except:
                sessions_info[session_id] = {
                    "status": "error",
                    "message": "Could not retrieve session data"
                }
        
        return jsonify({
            "active_sessions": len(active_sessions),
            "sessions": sessions_info,
            "redis_available": redis_client is not None,
            "orchestrator_available": ORCHESTRATOR_AVAILABLE,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Inventory Predictions Backend...")
    print(f"🔧 Enhanced Orchestrator: {'Available' if ORCHESTRATOR_AVAILABLE else 'Not Available'}")
    print(f"💾 Redis: {'Connected' if redis_client else 'Not Available'}")
    
    socketio.run(
        app, 
        host='0.0.0.0', 
        port=5000, 
        debug=True,
        allow_unsafe_werkzeug=True
    ) 