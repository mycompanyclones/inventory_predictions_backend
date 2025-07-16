# """
# Enhanced Session Manager for Supply Chain Management
# ===================================================

# This module provides comprehensive session management with:
# 1. Redis-based persistent storage
# 2. Complete conversation history tracking
# 3. Session context management
# 4. Tool result storage and retrieval
# 5. Enhanced session analytics
# """

# import json
# import time
# import pickle
# from datetime import datetime, timedelta
# from typing import Dict, List, Any, Optional
# import redis
# from agentic_system import SessionManager as BaseSessionManager

# class SessionManager(BaseSessionManager):
#     """
#     Enhanced session manager that extends the base functionality
#     with additional methods for comprehensive session management
#     """
    
#     def __init__(self, session_id: str):
#         """Initialize enhanced session manager"""
#         super().__init__(session_id)
#         self.context_key = f"session_context:{session_id}"
#         self.tools_key = f"session_tools:{session_id}"
        
#         # Initialize session context if not exists
#         if not self.redis_client.exists(self.context_key):
#             self._initialize_session_context()
    
#     def _initialize_session_context(self):
#         """Initialize session context with default values"""
#         context = {
#             'session_id': self.session_id,
#             'created_at': datetime.now().isoformat(),
#             'data_loaded': False,
#             'available_products': [],
#             'last_analysis_time': None,
#             'analysis_summary': {},
#             'user_preferences': {},
#             'query_patterns': []
#         }
        
#         self.redis_client.set(self.context_key, json.dumps(context).encode())
#         self.redis_client.expire(self.context_key, 1800)  # 30 minutes TTL
    
#     def add_to_history(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
#         """
#         Add message to conversation history
        
#         Args:
#             role: Role of the message sender ('user' or 'assistant')
#             content: Message content
#             metadata: Additional metadata for the message
#         """
#         message = {
#             'role': role,
#             'content': content,
#             'timestamp': datetime.now().isoformat()
#         }
        
#         if metadata:
#             message['metadata'] = metadata
        
#         self.add_message_to_history(message)
    
#     def add_tool_result(self, tool_name: str, result_data: Any, execution_time: Optional[float] = None):
#         """
#         Add tool execution result to session
        
#         Args:
#             tool_name: Name of the executed tool
#             result_data: Result data from tool execution
#             execution_time: Time taken to execute the tool
#         """
#         tool_result = {
#             'tool_name': tool_name,
#             'result_data': result_data,
#             'execution_time': execution_time,
#             'timestamp': datetime.now().isoformat()
#         }
        
#         # Store in Redis with proper encoding
#         self.redis_client.lpush(self.tools_key, json.dumps(tool_result).encode())
#         self.redis_client.expire(self.tools_key, 1800)  # 30 minutes TTL
        
#         # Limit to last 50 tool results
#         self.redis_client.ltrim(self.tools_key, 0, 49)
    
#     def get_tool_results(self) -> List[Dict[str, Any]]:
#         """Get all tool results for this session"""
#         tools = self.redis_client.lrange(self.tools_key, 0, -1)
#         return [json.loads(tool.decode() if isinstance(tool, bytes) else tool) for tool in tools]
    
#     def get_session_context(self) -> Dict[str, Any]:
#         """Get current session context"""
#         context_data = self.redis_client.get(self.context_key)
#         if context_data:
#             return json.loads(context_data)
#         return {}
    
#     def update_context(self, updates: Dict[str, Any]):
#         """
#         Update session context with new information
        
#         Args:
#             updates: Dictionary of context updates
#         """
#         current_context = self.get_session_context()
#         current_context.update(updates)
#         current_context['last_updated'] = datetime.now().isoformat()
        
#         self.redis_client.set(self.context_key, json.dumps(current_context).encode())
#         self.redis_client.expire(self.context_key, 1800)  # Reset TTL
    
#     def get_conversation_history(self) -> List[Any]:
#         """Get complete conversation history"""
#         message_history = self.get_message_history()
#         return message_history
    
#     def clear_history(self):
#         """Clear conversation history and reset session"""
#         # Clear message history
#         data = self._get_session_data()
#         if data:
#             data['message_history'] = []
#             data['query_count'] = 0
#             data['total_processing_time'] = 0.0
#             data['last_activity'] = datetime.now().isoformat()
#             self._save_session_data(data)
        
#         # Clear tool results
#         self.redis_client.delete(self.tools_key)
        
#         # Reset session context
#         self._initialize_session_context()
    
#     def finalize_session(self):
#         """Finalize session and save final state"""
#         data = self._get_session_data()
#         if data:
#             data['session_end'] = datetime.now().isoformat()
#             data['final_query_count'] = data.get('query_count', 0)
#             data['total_session_time'] = (datetime.now() - datetime.fromisoformat(data['session_start'])).total_seconds()
#             self._save_session_data(data)
            
#             # Extend TTL for archived sessions
#             self.redis_client.expire(self.redis_key, 3600)  # 1 hour for archived sessions
    
#     def get_session_info(self) -> Dict[str, Any]:
#         """Get comprehensive session information"""
#         session_data = self._get_session_data()
#         context_data = self.get_session_context()
#         tool_results = self.get_tool_results()
        
#         if not session_data:
#             return {
#                 'session_id': self.session_id,
#                 'status': 'not_found',
#                 'error': 'Session data not found'
#             }
        
#         return {
#             'session_id': self.session_id,
#             'status': 'active',
#             'created_at': session_data.get('session_start'),
#             'last_activity': session_data.get('last_activity'),
#             'query_count': session_data.get('query_count', 0),
#             'total_processing_time': session_data.get('total_processing_time', 0),
#             'message_history_length': len(session_data.get('message_history', [])),
#             'tool_results_count': len(tool_results),
#             'context_data': context_data,
#             'session_duration': (datetime.now() - datetime.fromisoformat(session_data['session_start'])).total_seconds(),
#             'data_loaded': context_data.get('data_loaded', False),
#             'available_products': context_data.get('available_products', []),
#             'last_analysis_time': context_data.get('last_analysis_time'),
#             'analysis_summary': context_data.get('analysis_summary', {})
#         }
    
#     def get_session_analytics(self) -> Dict[str, Any]:
#         """Get session analytics and insights"""
#         session_data = self._get_session_data()
#         tool_results = self.get_tool_results()
        
#         if not session_data:
#             return {'error': 'Session not found'}
        
#         # Calculate analytics
#         message_history = session_data.get('message_history', [])
#         user_messages = [msg for msg in message_history if msg.get('role') == 'user']
#         assistant_messages = [msg for msg in message_history if msg.get('role') == 'assistant']
        
#         # Tool usage analytics
#         tool_usage = {}
#         for tool_result in tool_results:
#             tool_name = tool_result.get('tool_name')
#             if tool_name:
#                 tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
        
#         # Query patterns
#         query_patterns = []
#         for msg in user_messages:
#             content = msg.get('content', '').lower()
#             if 'forecast' in content or 'predict' in content:
#                 query_patterns.append('forecasting')
#             elif 'stock' in content or 'inventory' in content:
#                 query_patterns.append('inventory')
#             elif 'risk' in content or 'stockout' in content:
#                 query_patterns.append('risk_assessment')
#             elif 'reorder' in content or 'order' in content:
#                 query_patterns.append('reordering')
        
#         return {
#             'session_id': self.session_id,
#             'total_queries': len(user_messages),
#             'total_responses': len(assistant_messages),
#             'avg_query_length': sum(len(msg.get('content', '')) for msg in user_messages) / max(1, len(user_messages)),
#             'avg_response_length': sum(len(msg.get('content', '')) for msg in assistant_messages) / max(1, len(assistant_messages)),
#             'tool_usage': tool_usage,
#             'most_used_tool': max(tool_usage.items(), key=lambda x: x[1]) if tool_usage else None,
#             'query_patterns': query_patterns,
#             'session_duration': (datetime.now() - datetime.fromisoformat(session_data['session_start'])).total_seconds(),
#             'avg_processing_time': session_data.get('total_processing_time', 0) / max(1, session_data.get('query_count', 1)),
#             'total_processing_time': session_data.get('total_processing_time', 0)
#         }
    
#     def add_user_preference(self, key: str, value: Any):
#         """Add user preference to session context"""
#         context = self.get_session_context()
#         if 'user_preferences' not in context:
#             context['user_preferences'] = {}
        
#         context['user_preferences'][key] = value
#         self.update_context(context)
    
#     def get_user_preferences(self) -> Dict[str, Any]:
#         """Get user preferences from session context"""
#         context = self.get_session_context()
#         return context.get('user_preferences', {})
    
#     def log_query_pattern(self, query: str, pattern_type: str):
#         """Log query pattern for analytics"""
#         context = self.get_session_context()
#         if 'query_patterns' not in context:
#             context['query_patterns'] = []
        
#         pattern_entry = {
#             'query': query[:100],  # Truncate for privacy
#             'pattern_type': pattern_type,
#             'timestamp': datetime.now().isoformat()
#         }
        
#         context['query_patterns'].append(pattern_entry)
        
#         # Keep only last 20 patterns
#         if len(context['query_patterns']) > 20:
#             context['query_patterns'] = context['query_patterns'][-20:]
        
#         self.update_context(context)
    
#     def get_recent_tools(self, limit: int = 5) -> List[Dict[str, Any]]:
#         """Get recently used tools"""
#         tools = self.get_tool_results()
#         return tools[:limit]
    
#     def get_session_summary(self) -> Dict[str, Any]:
#         """Get a summary of the session for display"""
#         session_info = self.get_session_info()
#         analytics = self.get_session_analytics()
        
#         return {
#             'session_id': self.session_id,
#             'status': session_info.get('status'),
#             'duration': session_info.get('session_duration'),
#             'queries_processed': session_info.get('query_count'),
#             'tools_used': analytics.get('tool_usage', {}),
#             'most_used_tool': analytics.get('most_used_tool'),
#             'query_patterns': analytics.get('query_patterns', []),
#             'data_loaded': session_info.get('data_loaded'),
#             'last_analysis': session_info.get('last_analysis_time'),
#             'avg_processing_time': analytics.get('avg_processing_time')
#         }
    
#     @property
#     def redis_client(self):
#         """Get Redis client instance"""
#         import redis
#         # Use the same client from the parent class
#         return redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    
#     def __repr__(self):
#         """String representation of session manager"""
#         return f"SessionManager(session_id='{self.session_id}')" 