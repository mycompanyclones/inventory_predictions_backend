# """
# System Validator for Restructured Backend
# =========================================

# This script validates the enhanced backend system to ensure:
# 1. All modules are properly imported
# 2. Redis connection is working
# 3. Tool registry is functional
# 4. Session management is working
# 5. Orchestrator can process queries
# """

# import sys
# import os
# import time
# import json
# from datetime import datetime
# from typing import Dict, Any, List

# def test_imports():
#     """Test that all modules can be imported"""
#     print("🔍 Testing module imports...")
    
#     try:
#         # Test basic imports
#         import redis
#         print("✅ Redis import successful")
        
#         import pandas as pd
#         print("✅ Pandas import successful")
        
#         import numpy as np
#         print("✅ NumPy import successful")
        
#         # Test custom module imports
#         try:
#             from agentic_system import DataTools, AnalyticsTools, BusinessTools, SessionManager as BaseSessionManager
#             print("✅ Agentic system imports successful")
#         except ImportError as e:
#             print(f"❌ Agentic system import failed: {e}")
#             return False
        
#         try:
#             from orchestration_system import ToolRegistry as BaseToolRegistry, ResponseSynthesizer, ExecutionContext
#             print("✅ Orchestration system imports successful")
#         except ImportError as e:
#             print(f"❌ Orchestration system import failed: {e}")
#             return False
        
#         try:
#             from enhanced_orchestrator import EnhancedOrchestrator
#             print("✅ Enhanced orchestrator import successful")
#         except ImportError as e:
#             print(f"❌ Enhanced orchestrator import failed: {e}")
#             return False
        
#         try:
#             from session_manager import SessionManager
#             print("✅ Session manager import successful")
#         except ImportError as e:
#             print(f"❌ Session manager import failed: {e}")
#             return False
        
#         try:
#             from tool_registry import ToolRegistry, ToolCategory
#             print("✅ Tool registry import successful")
#         except ImportError as e:
#             print(f"❌ Tool registry import failed: {e}")
#             return False
        
#         return True
        
#     except Exception as e:
#         print(f"❌ Import test failed: {e}")
#         return False

# def test_redis_connection():
#     """Test Redis connection"""
#     print("\n🔍 Testing Redis connection...")
    
#     try:
#         import redis
#         client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        
#         # Test basic operations
#         test_key = "test_key"
#         test_value = "test_value"
        
#         client.set(test_key, test_value)
#         retrieved_value = client.get(test_key)
        
#         if retrieved_value == test_value:
#             print("✅ Redis connection and operations successful")
#             client.delete(test_key)
#             return True
#         else:
#             print("❌ Redis value mismatch")
#             return False
            
#     except Exception as e:
#         print(f"❌ Redis connection failed: {e}")
#         print("💡 Make sure Redis is running: redis-server")
#         return False

# def test_session_manager():
#     """Test session manager functionality"""
#     print("\n🔍 Testing session manager...")
    
#     try:
#         from session_manager import SessionManager
        
#         # Create a test session
#         session_id = f"test_session_{int(time.time())}"
#         session_manager = SessionManager(session_id)
        
#         # Test adding messages
#         session_manager.add_message_to_history({"role": "user", "content": "Test message"})
#         session_manager.add_message_to_history({"role": "assistant", "content": "Test response"})
        
#         # Test getting history
#         history = session_manager.get_conversation_history()
        
#         if len(history) >= 2:
#             print("✅ Session manager message handling successful")
#         else:
#             print("❌ Session manager message handling failed")
#             return False
        
#         # Test session info
#         session_info = session_manager.get_session_info()
        
#         if session_info and session_info.get('session_id') == session_id:
#             print("✅ Session manager info retrieval successful")
#         else:
#             print("❌ Session manager info retrieval failed")
#             return False
        
#         # Clean up
#         session_manager.clear_history()
#         print("✅ Session manager cleanup successful")
        
#         return True
        
#     except Exception as e:
#         print(f"❌ Session manager test failed: {e}")
#         return False

# def test_tool_registry():
#     """Test tool registry functionality"""
#     print("\n🔍 Testing tool registry...")
    
#     try:
#         from tool_registry import ToolRegistry, ToolCategory
        
#         registry = ToolRegistry()
        
#         # Test getting all tools
#         tools_info = registry.get_all_tools_info()
        
#         if tools_info and 'tools' in tools_info:
#             print(f"✅ Tool registry loaded {tools_info['total_tools']} tools")
#         else:
#             print("❌ Tool registry failed to load tools")
#             return False
        
#         # Test tool categories
#         categories = [cat.value for cat in ToolCategory]
#         print(f"✅ Tool registry has {len(categories)} categories: {', '.join(categories)}")
        
#         # Test tool recommendation
#         recommendations = registry.get_recommended_tools("forecast demand")
        
#         if recommendations:
#             print(f"✅ Tool recommendations working: {', '.join(recommendations)}")
#         else:
#             print("❌ Tool recommendations failed")
#             return False
        
#         return True
        
#     except Exception as e:
#         print(f"❌ Tool registry test failed: {e}")
#         return False

# def test_enhanced_orchestrator():
#     """Test enhanced orchestrator functionality"""
#     print("\n🔍 Testing enhanced orchestrator...")
    
#     try:
#         from enhanced_orchestrator import EnhancedOrchestrator
        
#         orchestrator = EnhancedOrchestrator()
        
#         # Test query analysis
#         query = "What is the current inventory status?"
#         required_tools = orchestrator._analyze_query_requirements(query, [])
        
#         if required_tools:
#             print(f"✅ Query analysis successful: {len(required_tools)} tools identified")
#         else:
#             print("❌ Query analysis failed")
#             return False
        
#         # Test tool mapping
#         available_tools = list(orchestrator.tool_mapping.keys())
#         print(f"✅ Enhanced orchestrator has {len(available_tools)} mapped tools")
        
#         return True
        
#     except Exception as e:
#         print(f"❌ Enhanced orchestrator test failed: {e}")
#         return False

# def test_data_loading():
#     """Test data loading capability"""
#     print("\n🔍 Testing data loading...")
    
#     try:
#         import pandas as pd
        
#         # Check if supply chain data file exists
#         data_file = "supply_chain_data.csv"
        
#         if os.path.exists(data_file):
#             df = pd.read_csv(data_file)
#             print(f"✅ Data file loaded: {len(df)} records, {len(df.columns)} columns")
#             print(f"✅ Products in data: {df['product_name'].nunique()}")
#             return True
#         else:
#             print(f"❌ Data file {data_file} not found")
#             return False
            
#     except Exception as e:
#         print(f"❌ Data loading test failed: {e}")
#         return False

# def test_full_system_integration():
#     """Test full system integration"""
#     print("\n🔍 Testing full system integration...")
    
#     try:
#         from enhanced_orchestrator import EnhancedOrchestrator
#         from session_manager import SessionManager
        
#         # Create test session
#         session_id = f"integration_test_{int(time.time())}"
#         session_manager = SessionManager(session_id)
#         orchestrator = EnhancedOrchestrator()
        
#         # Test simple query processing
#         test_query = "Show me the current inventory status"
        
#         # Mock callbacks for testing
#         def mock_callback(*args, **kwargs):
#             pass
        
#         callbacks = {
#             'on_tool_start': mock_callback,
#             'on_tool_progress': mock_callback,
#             'on_tool_complete': mock_callback,
#             'on_synthesis_start': mock_callback,
#             'on_synthesis_progress': mock_callback,
#             'on_error': mock_callback
#         }
        
#         # Test basic orchestration
#         session_history = session_manager.get_conversation_history()
#         session_context = session_manager.get_session_context() if hasattr(session_manager, 'get_session_context') else {}
        
#         result = orchestrator.process_query_with_streaming(
#             user_query=test_query,
#             session_history=session_history,
#             session_context=session_context,
#             callbacks=callbacks
#         )
        
#         if result and 'response' in result:
#             print("✅ Full system integration successful")
#             print(f"✅ Response generated: {len(result['response'])} characters")
#             print(f"✅ Tools used: {len(result.get('tools_used', []))}")
#             return True
#         else:
#             print("❌ Full system integration failed")
#             return False
            
#     except Exception as e:
#         print(f"❌ Full system integration test failed: {e}")
#         return False

# def generate_system_report():
#     """Generate a comprehensive system report"""
#     print("\n📊 Generating system report...")
    
#     report = {
#         "validation_timestamp": datetime.now().isoformat(),
#         "system_version": "2.0 - Enhanced Orchestration System",
#         "tests_passed": 0,
#         "tests_total": 0,
#         "test_results": {},
#         "recommendations": []
#     }
    
#     # Run all tests
#     tests = [
#         ("imports", test_imports),
#         ("redis_connection", test_redis_connection),
#         ("session_manager", test_session_manager),
#         ("tool_registry", test_tool_registry),
#         ("enhanced_orchestrator", test_enhanced_orchestrator),
#         ("data_loading", test_data_loading),
#         ("full_system_integration", test_full_system_integration)
#     ]
    
#     for test_name, test_func in tests:
#         report["tests_total"] += 1
#         try:
#             result = test_func()
#             report["test_results"][test_name] = {
#                 "passed": result,
#                 "timestamp": datetime.now().isoformat()
#             }
#             if result:
#                 report["tests_passed"] += 1
#         except Exception as e:
#             report["test_results"][test_name] = {
#                 "passed": False,
#                 "error": str(e),
#                 "timestamp": datetime.now().isoformat()
#             }
    
#     # Generate recommendations
#     if report["test_results"]["redis_connection"]["passed"] is False:
#         report["recommendations"].append("Start Redis server: redis-server")
    
#     if report["test_results"]["data_loading"]["passed"] is False:
#         report["recommendations"].append("Ensure supply_chain_data.csv exists in the backend directory")
    
#     if report["tests_passed"] == report["tests_total"]:
#         report["status"] = "SYSTEM_READY"
#         report["message"] = "All tests passed. System is ready for production use."
#     elif report["tests_passed"] >= report["tests_total"] * 0.8:
#         report["status"] = "MOSTLY_READY"
#         report["message"] = "Most tests passed. System is mostly ready with minor issues."
#     else:
#         report["status"] = "NEEDS_ATTENTION"
#         report["message"] = "Multiple tests failed. System needs attention before use."
    
#     return report

# def main():
#     """Main validation function"""
#     print("🚀 Backend System Validation")
#     print("=" * 50)
    
#     # Generate and display report
#     report = generate_system_report()
    
#     print(f"\n📋 VALIDATION REPORT")
#     print("=" * 50)
#     print(f"Status: {report['status']}")
#     print(f"Message: {report['message']}")
#     print(f"Tests Passed: {report['tests_passed']}/{report['tests_total']}")
    
#     if report['recommendations']:
#         print("\n💡 Recommendations:")
#         for rec in report['recommendations']:
#             print(f"   - {rec}")
    
#     # Save report to file
#     report_file = f"validation_report_{int(time.time())}.json"
#     with open(report_file, 'w') as f:
#         json.dump(report, f, indent=2)
    
#     print(f"\n📄 Full report saved to: {report_file}")
    
#     # Return success status
#     return report['status'] == "SYSTEM_READY"

# if __name__ == "__main__":
#     success = main()
#     sys.exit(0 if success else 1) 