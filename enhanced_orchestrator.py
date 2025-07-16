"""
Enhanced Orchestrator for Supply Chain Management
================================================

This module provides a comprehensive orchestration system that:
1. Handles all types of queries with intelligent tool selection
2. Provides real-time streaming of tool execution
3. Integrates with session management for context awareness
4. Synthesizes comprehensive responses with markdown formatting
5. Supports comprehensive data visualization and analysis
"""

import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from openai import OpenAI
import os
from dotenv import load_dotenv
import traceback

# Import existing tools and systems
from agentic_system import DataTools, AnalyticsTools, BusinessTools
from orchestration_system import ToolRegistry, ResponseSynthesizer, ExecutionContext

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class EnhancedOrchestrator:
    """
    Enhanced orchestrator for supply chain management queries
    Handles all query types with comprehensive tool calling and real-time streaming
    """
    
    def __init__(self):
        self.data_tools = DataTools()
        self.analytics_tools = AnalyticsTools()
        self.business_tools = BusinessTools()
        self.tool_registry = ToolRegistry()
        self.response_synthesizer = ResponseSynthesizer()
        
        # Enhanced tool mapping for comprehensive coverage
        self.tool_mapping = {
            'load_data': self._load_supply_chain_data,
            'filter_data': self._filter_product_data,
            'analyze_inventory': self._analyze_inventory_status,
            'forecast_demand': self._calculate_demand_forecast,
            'assess_stockout_risk': self._analyze_stockout_risk,
            'generate_reorders': self._generate_reorder_recommendations,
            'analyze_trends': self._analyze_trends,
            'calculate_metrics': self._calculate_key_metrics,
            'visualize_data': self._create_data_visualization,
            'generate_insights': self._generate_insights,
            'comparative_analysis': self._perform_comparative_analysis,
            'optimization_recommendations': self._generate_optimization_recommendations
        }
    
    def process_query_with_streaming(self, 
                                   user_query: str,
                                   session_history: List[Any],
                                   session_context: Dict[str, Any],
                                   session_id: str) -> Dict[str, Any]:
        """
        Process a query with real-time streaming and comprehensive tool calling
        
        Args:
            user_query: The user's query
            session_history: Previous conversation history
            session_context: Current session context
            session_id: Session identifier
        
        Returns:
            Dict containing response and metadata
        """
        
        try:
            start_time = time.time()
            from agentic_system import Orchestrator
            from flask_socketio import emit
            from datetime import datetime
            
            result = Orchestrator.decide_tools_and_generate_insights(user_query, session_history, session_id, {})
            
            # Handle both new dict format and old string format for backward compatibility
            if isinstance(result, dict):
                return result
            else:
                # Old string format - wrap in new structure
                processing_time = time.time() - start_time
                return {
                    'response': result,
                    'processing_time': processing_time,
                    'tools_used': [],
                    'success': True
                }
                
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Critical error in orchestration: {str(e)}"
            print(f"❌ [ERROR] {error_msg}")
            
            return {
                'response': f"I apologize, but I encountered an error while processing your query: {str(e)}. Please try rephrasing your question.",
                'processing_time': processing_time,
                'tools_used': [],
                'success': False,
                'error': str(e)
            }
    
    def _analyze_query_requirements(self, user_query: str, session_history: List[Any]) -> List[Dict[str, Any]]:
        """
        Analyze the user query to determine which tools are needed
        
        Args:
            user_query: The user's query
            session_history: Previous conversation history
        
        Returns:
            List of required tools with their parameters
        """
        query_lower = user_query.lower()
        required_tools = []
        
        # Intent detection based on keywords and context
        intents = {
            'inventory_analysis': ['inventory', 'stock', 'current', 'available', 'levels'],
            'demand_forecasting': ['forecast', 'predict', 'demand', 'future', 'projection'],
            'stockout_risk': ['stockout', 'risk', 'shortage', 'out of stock', 'critical'],
            'reorder_recommendations': ['reorder', 'replenish', 'order', 'purchase', 'buy'],
            'trend_analysis': ['trend', 'pattern', 'historical', 'over time', 'change'],
            'comparative_analysis': ['compare', 'versus', 'vs', 'between', 'difference'],
            'optimization': ['optimize', 'best', 'improve', 'efficient', 'cost'],
            'visualization': ['chart', 'graph', 'plot', 'visual', 'show me']
        }
        
        detected_intents = []
        for intent, keywords in intents.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_intents.append(intent)
        
        # Default to comprehensive analysis if no specific intent
        if not detected_intents:
            detected_intents = ['inventory_analysis', 'demand_forecasting', 'stockout_risk']
        
        # Map intents to tools
        intent_to_tools = {
            'inventory_analysis': [
                {'name': 'analyze_inventory', 'description': 'Analyze current inventory status'},
                {'name': 'calculate_metrics', 'description': 'Calculate key inventory metrics'}
            ],
            'demand_forecasting': [
                {'name': 'forecast_demand', 'description': 'Generate demand forecasts'},
                {'name': 'analyze_trends', 'description': 'Analyze demand trends'}
            ],
            'stockout_risk': [
                {'name': 'assess_stockout_risk', 'description': 'Assess stockout risk for products'}
            ],
            'reorder_recommendations': [
                {'name': 'generate_reorders', 'description': 'Generate reorder recommendations'}
            ],
            'trend_analysis': [
                {'name': 'analyze_trends', 'description': 'Analyze historical trends'}
            ],
            'comparative_analysis': [
                {'name': 'comparative_analysis', 'description': 'Perform comparative analysis'}
            ],
            'optimization': [
                {'name': 'optimization_recommendations', 'description': 'Generate optimization recommendations'}
            ],
            'visualization': [
                {'name': 'visualize_data', 'description': 'Create data visualizations'}
            ]
        }
        
        # Add tools based on detected intents
        for intent in detected_intents:
            if intent in intent_to_tools:
                required_tools.extend(intent_to_tools[intent])
        
        # Always include insights generation
        required_tools.append({
            'name': 'generate_insights',
            'description': 'Generate actionable insights'
        })
        
        return required_tools
    
    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """
        Execute a specific tool with given parameters
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters for the tool
            context: Current execution context
        
        Returns:
            Tool execution results
        """
        if tool_name in self.tool_mapping:
            return self.tool_mapping[tool_name](parameters, context)
        else:
            # Fallback to existing tool registry
            return self.tool_registry.execute_tool(tool_name, parameters, context)
    
    # Enhanced Tool Implementations
    
    def _load_supply_chain_data(self, params: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Load supply chain data with enhanced metadata"""
        try:
            file_path = params.get('file_path', 'supply_chain_data.csv')
            df = pd.read_csv(file_path)
            
            # Store in context
            context.data_context['supply_chain_data'] = df
            context.data_context['data_loaded'] = True
            
            # Generate metadata
            products = df['product_name'].unique().tolist()
            date_range = {
                'start': df['date'].min(),
                'end': df['date'].max()
            }
            
            return {
                'success': True,
                'records_loaded': len(df),
                'products': products,
                'date_range': date_range,
                'columns': df.columns.tolist(),
                'data_summary': df.describe().to_dict()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'records_loaded': 0
            }
    
    def _filter_product_data(self, params: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Filter product data based on parameters"""
        try:
            df = context.data_context.get('supply_chain_data')
            if df is None:
                return {'success': False, 'error': 'No data loaded'}
            
            filtered_df = df.copy()
            
            # Apply filters
            if 'product_name' in params:
                filtered_df = filtered_df[filtered_df['product_name'] == params['product_name']]
            
            if 'date_start' in params:
                filtered_df = filtered_df[filtered_df['date'] >= params['date_start']]
            
            if 'date_end' in params:
                filtered_df = filtered_df[filtered_df['date'] <= params['date_end']]
            
            if 'min_stock' in params:
                filtered_df = filtered_df[filtered_df['current_stock'] >= params['min_stock']]
            
            if 'max_stock' in params:
                filtered_df = filtered_df[filtered_df['current_stock'] <= params['max_stock']]
            
            # Store filtered data
            context.data_context['filtered_data'] = filtered_df
            
            return {
                'success': True,
                'records_filtered': len(filtered_df),
                'filter_summary': params,
                'products_in_filter': filtered_df['product_name'].unique().tolist()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _analyze_inventory_status(self, params: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Analyze current inventory status"""
        try:
            df = context.data_context.get('supply_chain_data')
            if df is None:
                return {'success': False, 'error': 'No data loaded'}
            
            # Get latest data for each product
            latest_data = df.groupby('product_name').last()
            
            # Calculate inventory metrics
            inventory_analysis = {
                'total_products': len(latest_data),
                'total_stock_value': (latest_data['current_stock'] * latest_data['unit_cost']).sum(),
                'average_stock_level': latest_data['current_stock'].mean(),
                'low_stock_products': latest_data[latest_data['current_stock'] < 100].index.tolist(),
                'high_stock_products': latest_data[latest_data['current_stock'] > 1000].index.tolist(),
                'stock_distribution': latest_data['current_stock'].describe().to_dict()
            }
            
            return {
                'success': True,
                'inventory_analysis': inventory_analysis,
                'latest_data': latest_data.to_dict('index')
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _calculate_demand_forecast(self, params: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Calculate demand forecasts for products"""
        try:
            df = context.data_context.get('supply_chain_data')
            if df is None:
                return {'success': False, 'error': 'No data loaded'}
            
            forecast_days = params.get('forecast_days', 14)
            forecasts = {}
            
            for product in df['product_name'].unique():
                product_data = df[df['product_name'] == product]
                
                # Use analytics tools for forecasting
                forecast = self.analytics_tools.demand_forecaster(product_data, forecast_days)
                forecasts[product] = {
                    'forecast': forecast,
                    'total_demand': sum(forecast),
                    'average_daily_demand': sum(forecast) / len(forecast),
                    'confidence_level': params.get('confidence_level', 0.95)
                }
            
            return {
                'success': True,
                'forecasts': forecasts,
                'forecast_period': forecast_days,
                'forecast_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _analyze_stockout_risk(self, params: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Analyze stockout risk for products"""
        try:
            df = context.data_context.get('supply_chain_data')
            if df is None:
                return {'success': False, 'error': 'No data loaded'}
            
            # Get forecasts from context if available
            forecasts = context.analysis_results.get('forecast_demand', {}).get('forecasts', {})
            
            stockout_risks = {}
            
            for product in df['product_name'].unique():
                product_data = df[df['product_name'] == product].iloc[-1]
                
                # Get or generate forecast
                if product in forecasts:
                    forecast = forecasts[product]['forecast']
                else:
                    forecast = self.analytics_tools.demand_forecaster(
                        df[df['product_name'] == product], 14
                    )
                
                # Calculate risk
                risk_analysis = self.analytics_tools.stockout_risk_assessor(
                    forecast,
                    product_data['current_stock'],
                    product_data['lead_time_days']
                )
                
                stockout_risks[product] = risk_analysis
            
            return {
                'success': True,
                'stockout_risks': stockout_risks,
                'high_risk_products': [p for p, r in stockout_risks.items() if r['stockout_probability'] > 0.7],
                'analysis_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _generate_reorder_recommendations(self, params: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Generate reorder recommendations"""
        try:
            df = context.data_context.get('supply_chain_data')
            if df is None:
                return {'success': False, 'error': 'No data loaded'}
            
            # Get forecasts from context if available
            forecasts = context.analysis_results.get('forecast_demand', {}).get('forecasts', {})
            
            reorder_recommendations = {}
            
            for product in df['product_name'].unique():
                product_data = df[df['product_name'] == product].iloc[-1]
                
                # Get or generate forecast
                if product in forecasts:
                    forecast = forecasts[product]['forecast']
                else:
                    forecast = self.analytics_tools.demand_forecaster(
                        df[df['product_name'] == product], 14
                    )
                
                # Generate recommendation
                recommendation = self.business_tools.replenishment_advisor(
                    product_data['current_stock'],
                    forecast,
                    product_data['unit_cost']
                )
                
                reorder_recommendations[product] = {
                    **recommendation,
                    'current_stock': product_data['current_stock'],
                    'unit_cost': product_data['unit_cost'],
                    'supplier': product_data['supplier'],
                    'lead_time': product_data['lead_time_days']
                }
            
            return {
                'success': True,
                'reorder_recommendations': reorder_recommendations,
                'urgent_reorders': [p for p, r in reorder_recommendations.items() if r['reorder_recommended']],
                'recommendation_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _analyze_trends(self, params: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Analyze trends in the data"""
        try:
            df = context.data_context.get('supply_chain_data')
            if df is None:
                return {'success': False, 'error': 'No data loaded'}
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            trend_analysis = {}
            
            for product in df['product_name'].unique():
                product_data = df[df['product_name'] == product].sort_values('date')
                
                # Calculate trends
                demand_trend = product_data['daily_demand'].pct_change().mean()
                stock_trend = product_data['current_stock'].pct_change().mean()
                
                trend_analysis[product] = {
                    'demand_trend': demand_trend,
                    'stock_trend': stock_trend,
                    'demand_volatility': product_data['daily_demand'].std(),
                    'stock_volatility': product_data['current_stock'].std(),
                    'trend_direction': 'increasing' if demand_trend > 0 else 'decreasing'
                }
            
            return {
                'success': True,
                'trend_analysis': trend_analysis,
                'overall_demand_trend': df.groupby('date')['daily_demand'].sum().pct_change().mean(),
                'analysis_period': {
                    'start': df['date'].min().isoformat(),
                    'end': df['date'].max().isoformat()
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _calculate_key_metrics(self, params: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Calculate key supply chain metrics"""
        try:
            df = context.data_context.get('supply_chain_data')
            if df is None:
                return {'success': False, 'error': 'No data loaded'}
            
            # Calculate key metrics
            metrics = {
                'inventory_turnover': df['daily_demand'].sum() / df['current_stock'].mean(),
                'fill_rate': 1 - (df['daily_demand'] > df['current_stock']).mean(),
                'average_lead_time': df['lead_time_days'].mean(),
                'total_inventory_value': (df['current_stock'] * df['unit_cost']).sum(),
                'stockout_frequency': (df['current_stock'] == 0).mean(),
                'demand_variability': df['daily_demand'].std() / df['daily_demand'].mean()
            }
            
            return {
                'success': True,
                'key_metrics': metrics,
                'calculation_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _create_data_visualization(self, params: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Create data visualization information"""
        try:
            df = context.data_context.get('supply_chain_data')
            if df is None:
                return {'success': False, 'error': 'No data loaded'}
            
            # Generate visualization data
            visualization_data = {
                'inventory_levels': df.groupby('product_name')['current_stock'].last().to_dict(),
                'demand_patterns': df.groupby('product_name')['daily_demand'].mean().to_dict(),
                'supplier_distribution': df['supplier'].value_counts().to_dict(),
                'lead_time_analysis': df.groupby('product_name')['lead_time_days'].mean().to_dict(),
                'cost_analysis': df.groupby('product_name')['unit_cost'].mean().to_dict()
            }
            
            return {
                'success': True,
                'visualization_data': visualization_data,
                'chart_suggestions': [
                    'Inventory Levels by Product',
                    'Demand Patterns Over Time',
                    'Supplier Distribution',
                    'Lead Time Analysis',
                    'Cost Analysis by Product'
                ]
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _generate_insights(self, params: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Generate actionable insights from all analysis"""
        try:
            insights = []
            
            # Analyze all results in context
            for tool_name, result in context.analysis_results.items():
                if result.get('success'):
                    insights.extend(self._extract_insights_from_result(tool_name, result))
            
            # Prioritize insights
            prioritized_insights = self._prioritize_insights(insights)
            
            return {
                'success': True,
                'insights': prioritized_insights,
                'total_insights': len(insights),
                'high_priority_insights': [i for i in prioritized_insights if i['priority'] == 'high']
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _perform_comparative_analysis(self, params: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Perform comparative analysis between products or time periods"""
        try:
            df = context.data_context.get('supply_chain_data')
            if df is None:
                return {'success': False, 'error': 'No data loaded'}
            
            # Compare products
            product_comparison = {}
            products = df['product_name'].unique()
            
            for product in products:
                product_data = df[df['product_name'] == product]
                product_comparison[product] = {
                    'average_demand': product_data['daily_demand'].mean(),
                    'demand_variability': product_data['daily_demand'].std(),
                    'average_stock': product_data['current_stock'].mean(),
                    'average_cost': product_data['unit_cost'].mean(),
                    'lead_time': product_data['lead_time_days'].mean()
                }
            
            # Find best and worst performers
            best_performer = max(product_comparison.items(), key=lambda x: x[1]['average_demand'])
            worst_performer = min(product_comparison.items(), key=lambda x: x[1]['average_demand'])
            
            return {
                'success': True,
                'product_comparison': product_comparison,
                'best_performer': best_performer,
                'worst_performer': worst_performer,
                'comparison_metrics': ['demand', 'stock', 'cost', 'lead_time']
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _generate_optimization_recommendations(self, params: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Generate optimization recommendations"""
        try:
            recommendations = []
            
            # Analyze stockout risks
            stockout_results = context.analysis_results.get('assess_stockout_risk', {})
            if stockout_results.get('success'):
                high_risk_products = stockout_results.get('high_risk_products', [])
                if high_risk_products:
                    recommendations.append({
                        'type': 'stockout_prevention',
                        'priority': 'high',
                        'description': f"Urgent: Address stockout risk for {', '.join(high_risk_products[:3])}",
                        'action': 'Increase safety stock levels and expedite orders'
                    })
            
            # Analyze inventory levels
            inventory_results = context.analysis_results.get('analyze_inventory', {})
            if inventory_results.get('success'):
                low_stock = inventory_results.get('inventory_analysis', {}).get('low_stock_products', [])
                if low_stock:
                    recommendations.append({
                        'type': 'inventory_optimization',
                        'priority': 'medium',
                        'description': f"Optimize inventory for {len(low_stock)} low-stock products",
                        'action': 'Review reorder points and safety stock levels'
                    })
            
            # Add more optimization logic based on other results
            
            return {
                'success': True,
                'optimization_recommendations': recommendations,
                'total_recommendations': len(recommendations),
                'high_priority_count': len([r for r in recommendations if r['priority'] == 'high'])
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    # Helper methods
    
    def _generate_tool_summary(self, tool_name: str, result: Dict[str, Any]) -> str:
        """Generate a summary of tool execution results"""
        if not result.get('success'):
            return f"Tool execution failed: {result.get('error', 'Unknown error')}"
        
        summaries = {
            'load_data': f"Loaded {result.get('records_loaded', 0)} records for {len(result.get('products', []))} products",
            'analyze_inventory': f"Analyzed {result.get('inventory_analysis', {}).get('total_products', 0)} products with total value ${result.get('inventory_analysis', {}).get('total_stock_value', 0):,.2f}",
            'forecast_demand': f"Generated forecasts for {len(result.get('forecasts', {}))} products",
            'assess_stockout_risk': f"Assessed risk for {len(result.get('stockout_risks', {}))} products, {len(result.get('high_risk_products', []))} high-risk",
            'generate_reorders': f"Generated recommendations for {len(result.get('reorder_recommendations', {}))} products, {len(result.get('urgent_reorders', []))} urgent"
        }
        
        return summaries.get(tool_name, f"Tool {tool_name} completed successfully")
    
    def _extract_insights_from_result(self, tool_name: str, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract insights from tool execution results"""
        insights = []
        
        if tool_name == 'assess_stockout_risk':
            high_risk = result.get('high_risk_products', [])
            if high_risk:
                insights.append({
                    'type': 'risk_alert',
                    'priority': 'high',
                    'message': f"High stockout risk detected for {', '.join(high_risk[:3])}",
                    'action': 'Immediate reordering required'
                })
        
        if tool_name == 'analyze_inventory':
            inventory_analysis = result.get('inventory_analysis', {})
            low_stock = inventory_analysis.get('low_stock_products', [])
            if low_stock:
                insights.append({
                    'type': 'inventory_alert',
                    'priority': 'medium',
                    'message': f"Low stock levels for {len(low_stock)} products",
                    'action': 'Review reorder points'
                })
        
        return insights
    
    def _prioritize_insights(self, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize insights by importance"""
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        return sorted(insights, key=lambda x: priority_order.get(x['priority'], 0), reverse=True)
    
    def _synthesize_comprehensive_response(self, 
                                         user_query: str, 
                                         context: ExecutionContext, 
                                         session_history: List[Any],
                                         progress_callback: Callable) -> str:
        """Synthesize a comprehensive response from all analysis results"""
        
        progress_callback("Analyzing query context and results...")
        
        # Use existing ResponseSynthesizer with enhancements
        base_response = self.response_synthesizer.synthesize_response(context)
        
        progress_callback("Generating additional insights...")
        
        # Add enhanced insights
        insights = context.analysis_results.get('generate_insights', {})
        if insights.get('success'):
            high_priority_insights = insights.get('high_priority_insights', [])
            if high_priority_insights:
                base_response += "\n\n## 🚨 Critical Insights\n\n"
                for insight in high_priority_insights[:3]:
                    base_response += f"- **{insight['type'].replace('_', ' ').title()}**: {insight['message']}\n"
        
        progress_callback("Finalizing comprehensive response...")
        
        # Add footer with processing info
        base_response += f"\n\n---\n*Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        
        return base_response
    
    def _extract_data_insights(self, context: ExecutionContext) -> Dict[str, Any]:
        """Extract key data insights from execution context"""
        insights = {}
        
        # Extract key metrics
        if 'calculate_metrics' in context.analysis_results:
            metrics = context.analysis_results['calculate_metrics'].get('key_metrics', {})
            insights['key_metrics'] = metrics
        
        # Extract risk information
        if 'assess_stockout_risk' in context.analysis_results:
            risk_data = context.analysis_results['assess_stockout_risk']
            insights['risk_summary'] = {
                'high_risk_products': risk_data.get('high_risk_products', []),
                'total_products_analyzed': len(risk_data.get('stockout_risks', {}))
            }
        
        # Extract reorder information
        if 'generate_reorders' in context.analysis_results:
            reorder_data = context.analysis_results['generate_reorders']
            insights['reorder_summary'] = {
                'urgent_reorders': reorder_data.get('urgent_reorders', []),
                'total_recommendations': len(reorder_data.get('reorder_recommendations', {}))
            }
        
        return insights
    
    def _generate_context_updates(self, context: ExecutionContext) -> Dict[str, Any]:
        """Generate context updates for session management"""
        updates = {
            'data_loaded': True,
            'last_analysis_time': datetime.now().isoformat(),
            'available_products': [],
            'analysis_summary': {}
        }
        
        # Extract available products
        if 'supply_chain_data' in context.data_context:
            df = context.data_context['supply_chain_data']
            updates['available_products'] = df['product_name'].unique().tolist()
        
        # Summarize analysis results
        for tool_name, result in context.analysis_results.items():
            if result.get('success'):
                updates['analysis_summary'][tool_name] = {
                    'completed': True,
                    'timestamp': datetime.now().isoformat()
                }
        
        return updates 