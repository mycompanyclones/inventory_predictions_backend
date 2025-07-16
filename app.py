from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import json
import os
import redis
import time
from datetime import datetime
import traceback
import asyncio
from typing import Dict, Any, Optional
import uuid

# Import our enhanced orchestration system
try:
    from enhanced_orchestrator import EnhancedOrchestrator
    from agentic_system import SessionManager
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    # Fallback to basic system if enhanced orchestrator not available
    from agentic_system import SessionManager, DataTools, AnalyticsTools, BusinessTools
    ORCHESTRATOR_AVAILABLE = False

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
CORS(app)

# Initialize SocketIO with CORS  
socketio = SocketIO(
    app, 
    cors_allowed_origins=["http://localhost:3000", "http://127.0.0.1:3000"], 
    async_mode='threading',
    logger=True,
    engineio_logger=True
)

# Redis connection for session management
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
    print("✅ Redis connected successfully")
except:
    redis_client = None
    print("⚠️  Redis not available, using in-memory sessions")

# Store active sessions
active_sessions = {}

# Initialize orchestrator
if ORCHESTRATOR_AVAILABLE:
    orchestrator = EnhancedOrchestrator()
    print("✅ Enhanced Orchestrator initialized")
else:
    # Fallback tools
    data_tools = DataTools()
    analytics_tools = AnalyticsTools()
    business_tools = BusinessTools()
    print("⚠️  Using fallback tools - Enhanced Orchestrator not available")

@app.route('/')
def index():
    return jsonify({
        "message": "Supply Chain Inventory Prediction API",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0 - Enhanced Orchestration System",
        "orchestrator_available": ORCHESTRATOR_AVAILABLE
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        if redis_client:
            redis_client.ping()
            redis_status = "connected"
        else:
            redis_status = "not available"
    except:
        redis_status = "disconnected"
    
    return jsonify({
        "status": "healthy",
        "redis": redis_status,
        "active_sessions": len(active_sessions),
        "orchestrator_available": ORCHESTRATOR_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    })

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    session_id = request.sid
    print(f"🔌 Client connected: {session_id}")
    
    # Initialize session manager
    session_manager = SessionManager(session_id)
    active_sessions[session_id] = {
        "manager": session_manager,
        "connected_at": datetime.now(),
        "last_activity": datetime.now()
    }
    
    # Send welcome message
    emit('message', {
        'type': 'welcome',
        'content': """# Welcome to Supply Chain Predictions 🚀

Your enhanced AI assistant for supply chain operations. I can help you with:
- **Inventory Analysis** - Current stock levels and trends
- **Demand Forecasting** - Predict future demand patterns
- **Stockout Risk Assessment** - Identify potential stockouts
- **Reorder Recommendations** - Optimal replenishment strategies
- **Data Visualization** - Interactive charts and insights

Ask me anything about your supply chain data!""",
        'session_id': session_id,
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    session_id = request.sid
    print(f"🔌 Client disconnected: {session_id}")
    
    # Clean up session
    if session_id in active_sessions:
        del active_sessions[session_id]

@socketio.on('messages')
def handle_messages(data):
    """
    Enhanced message handler with comprehensive orchestration
    Handles all types of queries with real-time processing feedback
    """
    try:
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id', request.sid)
        
        if not user_message:
            emit('error', {
                'type': 'validation_error',
                'message': 'Message cannot be empty',
                'session_id': session_id,
                'timestamp': datetime.now().isoformat()
            })
            return

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

        print(f"📨 Processing query [{session_id}]: {user_message[:100]}...")
        
        # Add user message to history
        session_manager.add_message_to_history({"role": "user", "content": user_message})
        
        # Process the query using orchestrator (emits in real-time)
        try:
            if ORCHESTRATOR_AVAILABLE:
                # Use enhanced orchestrator with real-time emissions
                result = process_with_orchestrator(user_message, session_manager, session_id)
                print("result from orchestrator===============================================", result)
            else:
                # Use fallback processing
                result = process_with_fallback(user_message, session_manager)
                
                # For fallback, still emit manually since it doesn't have real-time emissions
                emit('processing_start', {
                    'session_id': session_id,
                    'message_id': f'processing-{int(time.time() * 1000)}',
                    'query': user_message,
                    'timestamp': datetime.now().isoformat()
                })
                
                response_content = result.get('response', '')
                
                # Add assistant response to history
                session_manager.add_message_to_history({"role": "assistant", "content": response_content})
                
                # Stream fallback response
                emit('message', {
                    'streaming_status': 'message_start',
                    'session_id': session_id,
                    'timestamp': datetime.now().isoformat()
                })
                
                words = response_content.split(' ')
                current_content = ''
                for i, word in enumerate(words):
                    current_content += word + ' '
                    emit('message', {
                        'streaming_status': 'message_continue',
                        'text_response': current_content.strip(),
                        'session_id': session_id,
                        'timestamp': datetime.now().isoformat()
                    })
                    time.sleep(0.02)
                
                emit('message', {
                    'streaming_status': 'end',
                    'text_response': response_content,
                    'session_id': session_id,
                    'timestamp': datetime.now().isoformat()
                })
                
                emit('processing_complete', {
                    'session_id': session_id,
                    'success': True,
                    'timestamp': datetime.now().isoformat()
                })
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(f"❌ [ERROR] {error_msg}")
            print(f"🔍 [TRACEBACK] {traceback.format_exc()}")
            
            # Add error to history
            session_manager.add_message_to_history({"role": "assistant", "content": f"Error: {error_msg}"})
            
            # Emit error
            emit('error', {
                'session_id': session_id,
                'error_type': 'processing_error',
                'message': error_msg,
                'timestamp': datetime.now().isoformat()
            })
            
            # Emit processing complete with error
            emit('processing_complete', {
                'session_id': session_id,
                'success': False,
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            })
            
    except Exception as e:
        error_msg = f"Critical error in message handler: {str(e)}"
        print(f"❌ [CRITICAL ERROR] {error_msg}")
        print(f"🔍 [TRACEBACK] {traceback.format_exc()}")
        
        emit('error', {
            'session_id': session_id if 'session_id' in locals() else 'unknown',
            'error_type': 'critical_error',
            'message': error_msg,
            'timestamp': datetime.now().isoformat()
        })

def process_with_orchestrator(user_message: str, session_manager: SessionManager, session_id: str) -> Dict[str, Any]:
    """Process query using the enhanced orchestrator"""
    
    # # Define streaming callbacks for real-time updates
    # def on_tool_start(tool_name, description, parameters=None):
    #     """Emit tool start with detailed information"""
    #     emit('tool_start', {
    #         'session_id': session_id,
    #         'tool_name': tool_name,
    #         'description': description,
    #         'parameters': parameters,
    #         'timestamp': datetime.now().isoformat(),
    #         'status': 'running'
    #     })
    
    # def on_tool_progress(tool_name, progress_info):
    #     """Emit tool progress updates"""
    #     emit('tool_progress', {
    #         'session_id': session_id,
    #         'tool_name': tool_name,
    #         'progress': progress_info,
    #         'timestamp': datetime.now().isoformat()
    #     })
    
    # def on_tool_complete(tool_name, summary, result_data=None, execution_time=None):
    #     """Emit tool completion with results"""
    #     emit('tool_complete', {
    #         'session_id': session_id,
    #         'tool_name': tool_name,
    #         'summary': summary,
    #         'result_data': result_data,
    #         'execution_time': execution_time,
    #         'timestamp': datetime.now().isoformat(),
    #         'status': 'completed'
    #     })
    
    # def on_synthesis_start(synthesis_type):
    #     """Emit synthesis start"""
    #     emit('synthesis_start', {
    #         'session_id': session_id,
    #         'synthesis_type': synthesis_type,
    #         'timestamp': datetime.now().isoformat()
    #     })
    
    # def on_synthesis_progress(partial_response):
    #     """Emit partial synthesis results"""
    #     emit('synthesis_progress', {
    #         'session_id': session_id,
    #         'partial_response': partial_response,
    #         'timestamp': datetime.now().isoformat()
    #     })
    
    # def on_error(error_type, error_message, details=None):
    #     """Emit error information"""
    #     emit('error', {
    #         'session_id': session_id,
    #         'error_type': error_type,
    #         'message': error_message,
    #         'details': details,
    #         'timestamp': datetime.now().isoformat()
    #     })
    
    # Process the query through enhanced orchestrator
    result = orchestrator.process_query_with_streaming(
        user_query=user_message,
        session_history=session_manager.get_message_history(),
        session_context=getattr(session_manager, 'get_session_context', lambda: {})(),
        session_id=session_id
    )
    
    return result

def process_with_fallback(user_message: str, session_manager: SessionManager) -> Dict[str, Any]:
    """Fallback processing when orchestrator is not available"""
    
    response = f"""# Supply Chain Analysis Report

## Executive Summary
The analysis for your query "{user_message}" could not be completed due to a lack of the enhanced orchestration system. 

## Insights and Recommendations

Since the enhanced orchestrator is not available, please ensure:

1. **System Dependencies**: All required modules are properly installed
2. **Data Availability**: Supply chain data file is accessible
3. **Configuration**: Environment variables are properly set

## Summary of Analysis Results

| Analysis Task | Success | Error |
|---------------|---------|-------|
| Load Data | No | Orchestrator not available |
| Process Query | No | Orchestrator not available |
| Generate Insights | No | Orchestrator not available |

## Execution Summary

- Total Steps: 0
- Tools Executed: 0
- Successful Tools: 0

## Next Steps

1. Check system configuration and dependencies
2. Restart the application with proper orchestrator setup
3. Verify all required files are present

*Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    return {
        'response': response,
        'processing_time': 0.1,
        'tools_used': [],
        'success': False,
        'error': 'Enhanced orchestrator not available'
    }

@socketio.on('get_session_info')
def handle_get_session_info(data):
    """Get session information and statistics"""
    session_id = data.get('session_id', request.sid)
    
    if session_id in active_sessions:
        session_manager = active_sessions[session_id]["manager"]
        session_data = session_manager._get_session_data()
        
        session_info = {
            'session_id': session_id,
            'query_count': session_data.get('query_count', 0),
            'total_processing_time': session_data.get('total_processing_time', 0),
            'last_activity': session_data.get('last_activity'),
            'message_count': len(session_data.get('message_history', []))
        }
        
        emit('session_info', {
            'session_id': session_id,
            'info': session_info,
            'timestamp': datetime.now().isoformat()
        })
    else:
        emit('error', {
            'session_id': session_id,
            'error_type': 'session_not_found',
            'message': 'Session not found',
            'timestamp': datetime.now().isoformat()
        })

@socketio.on('clear_session')
def handle_clear_session(data):
    """Clear session history"""
    session_id = data.get('session_id', request.sid)
    
    if session_id in active_sessions:
        # Create new session to clear history
        session_manager = SessionManager(session_id)
        active_sessions[session_id]["manager"] = session_manager
        
        emit('session_cleared', {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat()
        })
    else:
        emit('error', {
            'session_id': session_id,
            'error_type': 'session_not_found',
            'message': 'Session not found',
            'timestamp': datetime.now().isoformat()
        })

# Admin endpoints
@app.route('/admin/sessions')
def admin_sessions():
    """Get all active sessions information"""
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
                "connected_at": session_data["connected_at"].isoformat(),
                "last_activity": session_data["last_activity"].isoformat(),
                "status": "error_retrieving_data"
            }
    
    return jsonify({
        "active_sessions": len(active_sessions),
        "sessions": sessions_info,
        "orchestrator_available": ORCHESTRATOR_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 Starting Enhanced Supply Chain Prediction API...")
    print("📊 Features: Real-time orchestration, Redis session management, comprehensive tool calling")
    print(f"🔧 Enhanced Orchestrator: {'Available' if ORCHESTRATOR_AVAILABLE else 'Not Available'}")
    print("🔗 Frontend should connect to: http://localhost:5000")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)