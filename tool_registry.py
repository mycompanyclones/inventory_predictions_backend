"""
Comprehensive Tool Registry for Supply Chain Management
======================================================

This module provides a centralized registry of all available tools with:
1. Tool metadata and descriptions
2. Parameter validation
3. Tool execution capabilities
4. Category organization
5. Performance monitoring
"""

import json
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

# Import existing tools
from agentic_system import DataTools, AnalyticsTools, BusinessTools
from orchestration_system import ToolRegistry as BaseToolRegistry

class ToolCategory(Enum):
    """Categories of available tools"""
    DATA_MANAGEMENT = "data_management"
    ANALYTICS = "analytics"
    BUSINESS_LOGIC = "business_logic"
    VISUALIZATION = "visualization"
    REPORTING = "reporting"
    OPTIMIZATION = "optimization"

@dataclass
class ToolDefinition:
    """Definition of a tool with metadata"""
    name: str
    description: str
    category: ToolCategory
    parameters: Dict[str, Any]
    returns: Dict[str, Any]
    execution_time_avg: Optional[float] = None
    success_rate: Optional[float] = None
    dependencies: List[str] = None
    examples: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.examples is None:
            self.examples = []

@dataclass
class ToolExecutionStats:
    """Statistics for tool execution"""
    tool_name: str
    total_executions: int
    successful_executions: int
    failed_executions: int
    average_execution_time: float
    last_execution: Optional[datetime] = None

class ToolRegistry:
    """
    Comprehensive tool registry that manages all available tools
    """
    
    def __init__(self):
        self.data_tools = DataTools()
        self.analytics_tools = AnalyticsTools()
        self.business_tools = BusinessTools()
        self.base_registry = BaseToolRegistry()
        
        # Initialize tool definitions
        self.tool_definitions = self._initialize_tool_definitions()
        
        # Execution statistics
        self.execution_stats: Dict[str, ToolExecutionStats] = {}
        
        # Performance monitoring
        self.performance_history: List[Dict[str, Any]] = []
    
    def _initialize_tool_definitions(self) -> Dict[str, ToolDefinition]:
        """Initialize all tool definitions with metadata"""
        
        definitions = {}
        
        # Data Management Tools
        definitions['load_supply_chain_data'] = ToolDefinition(
            name='load_supply_chain_data',
            description='Load supply chain data from CSV file into analysis context',
            category=ToolCategory.DATA_MANAGEMENT,
            parameters={
                'file_path': {
                    'type': 'string',
                    'description': 'Path to the CSV file',
                    'default': 'supply_chain_data.csv',
                    'required': False
                }
            },
            returns={
                'success': 'boolean',
                'records_loaded': 'integer',
                'products': 'array',
                'date_range': 'object',
                'columns': 'array',
                'data_summary': 'object'
            },
            examples=[
                {
                    'input': {'file_path': 'supply_chain_data.csv'},
                    'output': {'success': True, 'records_loaded': 54, 'products': ['Product A', 'Product B']}
                }
            ]
        )
        
        definitions['filter_product_data'] = ToolDefinition(
            name='filter_product_data',
            description='Filter supply chain data by product, date range, or stock levels',
            category=ToolCategory.DATA_MANAGEMENT,
            parameters={
                'product_name': {
                    'type': 'string',
                    'description': 'Name of the product to filter by',
                    'required': False
                },
                'product_id': {
                    'type': 'string',
                    'description': 'Product ID to filter by',
                    'required': False
                },
                'date_start': {
                    'type': 'string',
                    'description': 'Start date for filtering (YYYY-MM-DD)',
                    'required': False
                },
                'date_end': {
                    'type': 'string',
                    'description': 'End date for filtering (YYYY-MM-DD)',
                    'required': False
                },
                'min_stock': {
                    'type': 'number',
                    'description': 'Minimum stock level to filter by',
                    'required': False
                },
                'max_stock': {
                    'type': 'number',
                    'description': 'Maximum stock level to filter by',
                    'required': False
                }
            },
            returns={
                'success': 'boolean',
                'records_filtered': 'integer',
                'filter_summary': 'object',
                'products_in_filter': 'array'
            },
            dependencies=['load_supply_chain_data']
        )
        
        # Analytics Tools
        definitions['calculate_demand_forecast'] = ToolDefinition(
            name='calculate_demand_forecast',
            description='Generate demand forecasts for products using time series analysis',
            category=ToolCategory.ANALYTICS,
            parameters={
                'product_name': {
                    'type': 'string',
                    'description': 'Product to forecast demand for',
                    'required': False
                },
                'forecast_days': {
                    'type': 'integer',
                    'description': 'Number of days to forecast',
                    'default': 14,
                    'required': False
                },
                'confidence_level': {
                    'type': 'number',
                    'description': 'Confidence level for forecast (0.8-0.99)',
                    'default': 0.95,
                    'required': False
                }
            },
            returns={
                'success': 'boolean',
                'forecasts': 'object',
                'forecast_period': 'integer',
                'forecast_date': 'string'
            },
            dependencies=['load_supply_chain_data']
        )
        
        definitions['analyze_stockout_risk'] = ToolDefinition(
            name='analyze_stockout_risk',
            description='Analyze the risk of stockout for products based on current stock and demand patterns',
            category=ToolCategory.ANALYTICS,
            parameters={
                'product_name': {
                    'type': 'string',
                    'description': 'Product to analyze stockout risk for',
                    'required': False
                },
                'forecast_data': {
                    'type': 'array',
                    'description': 'Forecast demand data array',
                    'items': {'type': 'number'},
                    'required': False
                }
            },
            returns={
                'success': 'boolean',
                'stockout_risks': 'object',
                'high_risk_products': 'array',
                'analysis_date': 'string'
            },
            dependencies=['load_supply_chain_data', 'calculate_demand_forecast']
        )
        
        definitions['analyze_trends'] = ToolDefinition(
            name='analyze_trends',
            description='Analyze trends in demand, stock levels, and other key metrics',
            category=ToolCategory.ANALYTICS,
            parameters={
                'product_name': {
                    'type': 'string',
                    'description': 'Product to analyze trends for',
                    'required': False
                },
                'time_period': {
                    'type': 'string',
                    'description': 'Time period for trend analysis',
                    'default': 'all',
                    'required': False
                }
            },
            returns={
                'success': 'boolean',
                'trend_analysis': 'object',
                'overall_demand_trend': 'number',
                'analysis_period': 'object'
            },
            dependencies=['load_supply_chain_data']
        )
        
        # Business Logic Tools
        definitions['generate_reorder_recommendations'] = ToolDefinition(
            name='generate_reorder_recommendations',
            description='Generate optimal reorder recommendations based on stock levels and forecasts',
            category=ToolCategory.BUSINESS_LOGIC,
            parameters={
                'product_name': {
                    'type': 'string',
                    'description': 'Product to generate recommendations for',
                    'required': False
                },
                'current_stock': {
                    'type': 'number',
                    'description': 'Current stock level',
                    'required': False
                },
                'forecast_data': {
                    'type': 'array',
                    'description': 'Forecast demand data',
                    'items': {'type': 'number'},
                    'required': False
                },
                'safety_stock_factor': {
                    'type': 'number',
                    'description': 'Safety stock factor (0.1-0.5)',
                    'default': 0.1,
                    'required': False
                }
            },
            returns={
                'success': 'boolean',
                'reorder_recommendations': 'object',
                'urgent_reorders': 'array',
                'recommendation_date': 'string'
            },
            dependencies=['load_supply_chain_data', 'calculate_demand_forecast']
        )
        
        definitions['calculate_key_metrics'] = ToolDefinition(
            name='calculate_key_metrics',
            description='Calculate key supply chain performance metrics',
            category=ToolCategory.ANALYTICS,
            parameters={
                'metric_types': {
                    'type': 'array',
                    'description': 'Types of metrics to calculate',
                    'items': {'type': 'string'},
                    'default': ['all'],
                    'required': False
                }
            },
            returns={
                'success': 'boolean',
                'key_metrics': 'object',
                'calculation_date': 'string'
            },
            dependencies=['load_supply_chain_data']
        )
        
        # Visualization Tools
        definitions['create_data_visualization'] = ToolDefinition(
            name='create_data_visualization',
            description='Create data visualization information for charts and graphs',
            category=ToolCategory.VISUALIZATION,
            parameters={
                'chart_type': {
                    'type': 'string',
                    'description': 'Type of chart to create',
                    'default': 'multi',
                    'required': False
                },
                'products': {
                    'type': 'array',
                    'description': 'Products to include in visualization',
                    'items': {'type': 'string'},
                    'required': False
                }
            },
            returns={
                'success': 'boolean',
                'visualization_data': 'object',
                'chart_suggestions': 'array'
            },
            dependencies=['load_supply_chain_data']
        )
        
        # Optimization Tools
        definitions['generate_optimization_recommendations'] = ToolDefinition(
            name='generate_optimization_recommendations',
            description='Generate optimization recommendations for supply chain operations',
            category=ToolCategory.OPTIMIZATION,
            parameters={
                'optimization_type': {
                    'type': 'string',
                    'description': 'Type of optimization to perform',
                    'default': 'general',
                    'required': False
                },
                'priority_level': {
                    'type': 'string',
                    'description': 'Priority level for recommendations',
                    'default': 'all',
                    'required': False
                }
            },
            returns={
                'success': 'boolean',
                'optimization_recommendations': 'array',
                'total_recommendations': 'integer',
                'high_priority_count': 'integer'
            },
            dependencies=['load_supply_chain_data']
        )
        
        # Reporting Tools
        definitions['generate_insights'] = ToolDefinition(
            name='generate_insights',
            description='Generate actionable insights from all analysis results',
            category=ToolCategory.REPORTING,
            parameters={
                'insight_types': {
                    'type': 'array',
                    'description': 'Types of insights to generate',
                    'items': {'type': 'string'},
                    'default': ['all'],
                    'required': False
                }
            },
            returns={
                'success': 'boolean',
                'insights': 'array',
                'total_insights': 'integer',
                'high_priority_insights': 'array'
            },
            dependencies=['load_supply_chain_data']
        )
        
        definitions['perform_comparative_analysis'] = ToolDefinition(
            name='perform_comparative_analysis',
            description='Perform comparative analysis between products or time periods',
            category=ToolCategory.ANALYTICS,
            parameters={
                'comparison_type': {
                    'type': 'string',
                    'description': 'Type of comparison to perform',
                    'default': 'products',
                    'required': False
                },
                'products': {
                    'type': 'array',
                    'description': 'Products to compare',
                    'items': {'type': 'string'},
                    'required': False
                }
            },
            returns={
                'success': 'boolean',
                'product_comparison': 'object',
                'best_performer': 'array',
                'worst_performer': 'array',
                'comparison_metrics': 'array'
            },
            dependencies=['load_supply_chain_data']
        )
        
        return definitions
    
    def get_tool_definition(self, tool_name: str) -> Optional[ToolDefinition]:
        """Get definition for a specific tool"""
        return self.tool_definitions.get(tool_name)
    
    def get_tools_by_category(self, category: ToolCategory) -> List[ToolDefinition]:
        """Get all tools in a specific category"""
        return [tool for tool in self.tool_definitions.values() if tool.category == category]
    
    def get_all_tools_info(self) -> Dict[str, Any]:
        """Get information about all available tools"""
        tools_info = {}
        
        for tool_name, tool_def in self.tool_definitions.items():
            # Get execution stats if available
            stats = self.execution_stats.get(tool_name)
            
            # Convert tool definition to dict and fix ToolCategory serialization
            tool_def_dict = asdict(tool_def)
            tool_def_dict['category'] = tool_def.category.value  # Convert enum to string
            
            tools_info[tool_name] = {
                'definition': tool_def_dict,
                'statistics': asdict(stats) if stats else None,
                'available': True,
                'last_updated': datetime.now().isoformat()
            }
        
        return {
            'total_tools': len(self.tool_definitions),
            'categories': [cat.value for cat in ToolCategory],
            'tools': tools_info,
            'registry_version': '2.0',
            'last_updated': datetime.now().isoformat()
        }
    
    def get_tool_dependencies(self, tool_name: str) -> List[str]:
        """Get dependencies for a specific tool"""
        tool_def = self.get_tool_definition(tool_name)
        return tool_def.dependencies if tool_def else []
    
    def validate_tool_parameters(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters for a tool"""
        tool_def = self.get_tool_definition(tool_name)
        
        if not tool_def:
            return {
                'valid': False,
                'errors': [f'Tool {tool_name} not found']
            }
        
        errors = []
        warnings = []
        
        # Check required parameters
        for param_name, param_def in tool_def.parameters.items():
            if param_def.get('required', False) and param_name not in parameters:
                errors.append(f'Required parameter {param_name} is missing')
        
        # Check parameter types
        for param_name, param_value in parameters.items():
            if param_name in tool_def.parameters:
                param_def = tool_def.parameters[param_name]
                expected_type = param_def.get('type')
                
                # Basic type checking
                if expected_type == 'string' and not isinstance(param_value, str):
                    errors.append(f'Parameter {param_name} should be a string')
                elif expected_type == 'number' and not isinstance(param_value, (int, float)):
                    errors.append(f'Parameter {param_name} should be a number')
                elif expected_type == 'integer' and not isinstance(param_value, int):
                    errors.append(f'Parameter {param_name} should be an integer')
                elif expected_type == 'boolean' and not isinstance(param_value, bool):
                    errors.append(f'Parameter {param_name} should be a boolean')
                elif expected_type == 'array' and not isinstance(param_value, list):
                    errors.append(f'Parameter {param_name} should be an array')
            else:
                warnings.append(f'Unknown parameter {param_name}')
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """Execute a tool with performance monitoring"""
        start_time = time.time()
        
        # Update execution stats
        if tool_name not in self.execution_stats:
            self.execution_stats[tool_name] = ToolExecutionStats(
                tool_name=tool_name,
                total_executions=0,
                successful_executions=0,
                failed_executions=0,
                average_execution_time=0.0
            )
        
        stats = self.execution_stats[tool_name]
        stats.total_executions += 1
        stats.last_execution = datetime.now()
        
        try:
            # Execute the tool using the base registry
            result = self.base_registry.execute_tool(tool_name, parameters, context)
            
            # Update success stats
            stats.successful_executions += 1
            execution_time = time.time() - start_time
            
            # Update average execution time
            stats.average_execution_time = (
                (stats.average_execution_time * (stats.successful_executions - 1) + execution_time) /
                stats.successful_executions
            )
            
            # Store performance data
            self.performance_history.append({
                'tool_name': tool_name,
                'execution_time': execution_time,
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'parameters': parameters
            })
            
            return result
            
        except Exception as e:
            # Update failure stats
            stats.failed_executions += 1
            execution_time = time.time() - start_time
            
            # Store performance data
            self.performance_history.append({
                'tool_name': tool_name,
                'execution_time': execution_time,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'parameters': parameters
            })
            
            return {
                'success': False,
                'error': str(e),
                'tool_name': tool_name,
                'execution_time': execution_time
            }
    
    def get_tool_performance_stats(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics for tools"""
        if tool_name:
            stats = self.execution_stats.get(tool_name)
            return asdict(stats) if stats else {'error': 'Tool not found'}
        
        # Return all stats
        return {
            'all_tools': {name: asdict(stats) for name, stats in self.execution_stats.items()},
            'total_executions': sum(stats.total_executions for stats in self.execution_stats.values()),
            'total_successful': sum(stats.successful_executions for stats in self.execution_stats.values()),
            'total_failed': sum(stats.failed_executions for stats in self.execution_stats.values()),
            'overall_success_rate': (
                sum(stats.successful_executions for stats in self.execution_stats.values()) /
                max(1, sum(stats.total_executions for stats in self.execution_stats.values()))
            ),
            'last_updated': datetime.now().isoformat()
        }
    
    def get_recommended_tools(self, query: str) -> List[str]:
        """Get recommended tools based on query analysis"""
        query_lower = query.lower()
        recommendations = []
        
        # Intent-based recommendations
        if any(keyword in query_lower for keyword in ['load', 'data', 'csv', 'import']):
            recommendations.append('load_supply_chain_data')
        
        if any(keyword in query_lower for keyword in ['forecast', 'predict', 'demand', 'future']):
            recommendations.extend(['calculate_demand_forecast', 'analyze_trends'])
        
        if any(keyword in query_lower for keyword in ['stock', 'inventory', 'current', 'levels']):
            recommendations.extend(['filter_product_data', 'calculate_key_metrics'])
        
        if any(keyword in query_lower for keyword in ['risk', 'stockout', 'shortage']):
            recommendations.append('analyze_stockout_risk')
        
        if any(keyword in query_lower for keyword in ['reorder', 'order', 'replenish', 'buy']):
            recommendations.append('generate_reorder_recommendations')
        
        if any(keyword in query_lower for keyword in ['compare', 'comparison', 'versus', 'vs']):
            recommendations.append('perform_comparative_analysis')
        
        if any(keyword in query_lower for keyword in ['optimize', 'optimization', 'improve']):
            recommendations.append('generate_optimization_recommendations')
        
        if any(keyword in query_lower for keyword in ['chart', 'graph', 'visual', 'plot']):
            recommendations.append('create_data_visualization')
        
        if any(keyword in query_lower for keyword in ['insight', 'summary', 'analysis']):
            recommendations.append('generate_insights')
        
        # Default recommendations if no specific intent detected
        if not recommendations:
            recommendations = [
                'load_supply_chain_data',
                'calculate_key_metrics',
                'generate_insights'
            ]
        
        return recommendations
    
    def get_tool_execution_order(self, tool_names: List[str]) -> List[str]:
        """Get optimal execution order for tools based on dependencies"""
        ordered_tools = []
        remaining_tools = tool_names.copy()
        
        while remaining_tools:
            # Find tools with no unresolved dependencies
            available_tools = []
            for tool_name in remaining_tools:
                tool_def = self.get_tool_definition(tool_name)
                if not tool_def:
                    continue
                
                # Check if all dependencies are satisfied
                dependencies_satisfied = all(
                    dep in ordered_tools for dep in tool_def.dependencies
                )
                
                if dependencies_satisfied:
                    available_tools.append(tool_name)
            
            if not available_tools:
                # Handle circular dependencies or missing dependencies
                # Add remaining tools in original order
                ordered_tools.extend(remaining_tools)
                break
            
            # Add available tools to ordered list
            for tool_name in available_tools:
                ordered_tools.append(tool_name)
                remaining_tools.remove(tool_name)
        
        return ordered_tools
    
    def clear_performance_history(self):
        """Clear performance history"""
        self.performance_history = []
        self.execution_stats = {}
    
    def export_tool_catalog(self) -> Dict[str, Any]:
        """Export complete tool catalog for documentation"""
        catalog = {
            'registry_info': {
                'version': '2.0',
                'total_tools': len(self.tool_definitions),
                'categories': [cat.value for cat in ToolCategory],
                'generated_at': datetime.now().isoformat()
            },
            'categories': {},
            'tools': {}
        }
        
        # Group tools by category
        for category in ToolCategory:
            catalog['categories'][category.value] = {
                'name': category.value,
                'tools': [tool.name for tool in self.get_tools_by_category(category)]
            }
        
        # Add detailed tool information
        for tool_name, tool_def in self.tool_definitions.items():
            catalog['tools'][tool_name] = asdict(tool_def)
        
        return catalog 