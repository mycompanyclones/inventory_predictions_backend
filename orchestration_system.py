"""
Advanced Tool Calling Orchestration System for Supply Chain Management
=====================================================================

This module implements a sophisticated orchestration system that:
1. Manages tool calling with recursive processing
2. Maintains execution context and memory
3. Synthesizes responses in markdown format
4. Provides beautiful data visualization capabilities
"""

import json
import time
import uuid
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# =====================================================================
# Core Data Structures
# =====================================================================

class ToolCallStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ToolCall:
    id: str
    name: str
    parameters: Dict[str, Any]
    status: ToolCallStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None

@dataclass
class ExecutionStep:
    step_id: str
    description: str
    tool_calls: List[ToolCall]
    reasoning: str
    created_at: datetime
    completed_at: Optional[datetime] = None

@dataclass
class ExecutionContext:
    session_id: str
    user_query: str
    steps: List[ExecutionStep]
    data_context: Dict[str, Any]
    analysis_results: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    is_complete: bool = False

# =====================================================================
# Tool Registry - All Available Supply Chain Tools
# =====================================================================

class ToolRegistry:
    """Registry of all available tools with their definitions and implementations"""
    
    @staticmethod
    def get_tool_definitions() -> List[Dict]:
        """Return OpenAI function calling definitions for all tools"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "load_supply_chain_data",
                    "description": "Load supply chain data from CSV file into analysis context",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the CSV file",
                                "default": "supply_chain_data.csv"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "filter_product_data",
                    "description": "Filter supply chain data by product name, date range, or other criteria",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_name": {
                                "type": "string",
                                "description": "Name of the product to filter by"
                            },
                            "product_id": {
                                "type": "string",
                                "description": "Product ID to filter by"
                            },
                            "date_start": {
                                "type": "string",
                                "description": "Start date for filtering (YYYY-MM-DD)"
                            },
                            "date_end": {
                                "type": "string", 
                                "description": "End date for filtering (YYYY-MM-DD)"
                            },
                            "min_stock": {
                                "type": "number",
                                "description": "Minimum stock level to filter by"
                            },
                            "max_stock": {
                                "type": "number",
                                "description": "Maximum stock level to filter by"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_demand_forecast",
                    "description": "Generate demand forecast for products using time series analysis",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_name": {
                                "type": "string",
                                "description": "Product to forecast demand for"
                            },
                            "forecast_days": {
                                "type": "integer",
                                "description": "Number of days to forecast",
                                "default": 14
                            },
                            "confidence_level": {
                                "type": "number",
                                "description": "Confidence level for forecast (0.8-0.99)",
                                "default": 0.95
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_stockout_risk",
                    "description": "Analyze the risk of stockout for products based on current stock and demand patterns",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_name": {
                                "type": "string",
                                "description": "Product to analyze stockout risk for"
                            },
                            "forecast_data": {
                                "type": "array",
                                "description": "Forecast demand data array",
                                "items": {"type": "number"}
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_reorder_recommendations",
                    "description": "Generate optimal reorder recommendations based on stock levels and forecasts",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_name": {
                                "type": "string",
                                "description": "Product to generate recommendations for"
                            },
                            "current_stock": {
                                "type": "number",
                                "description": "Current stock level"
                            },
                            "forecast_data": {
                                "type": "array",
                                "description": "Forecast demand data",
                                "items": {"type": "number"}
                            },
                            "lead_time_days": {
                                "type": "integer",
                                "description": "Lead time in days for reordering"
                            },
                            "safety_stock_ratio": {
                                "type": "number",
                                "description": "Safety stock ratio (0.1-0.5)",
                                "default": 0.2
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_inventory_metrics",
                    "description": "Calculate comprehensive inventory metrics including turnover, holding costs, etc.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_name": {
                                "type": "string",
                                "description": "Product to calculate metrics for"
                            },
                            "time_period_days": {
                                "type": "integer",
                                "description": "Time period for calculations",
                                "default": 30
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "compare_products",
                    "description": "Compare multiple products across various metrics",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_names": {
                                "type": "array",
                                "description": "List of product names to compare",
                                "items": {"type": "string"}
                            },
                            "metrics": {
                                "type": "array",
                                "description": "Metrics to compare",
                                "items": {"type": "string"},
                                "default": ["current_stock", "daily_demand", "stockout_risk"]
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_data_table",
                    "description": "Generate formatted data tables for analysis results",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "data_type": {
                                "type": "string",
                                "description": "Type of data table to generate",
                                "enum": ["inventory_summary", "forecast_results", "risk_analysis", "reorder_recommendations", "product_comparison"]
                            },
                            "include_charts": {
                                "type": "boolean",
                                "description": "Whether to include chart data",
                                "default": False
                            }
                        }
                    }
                }
            }
        ]
    
    @staticmethod
    def execute_tool(tool_call: ToolCall, context: ExecutionContext) -> Any:
        """Execute a specific tool call and return results"""
        tool_name = tool_call.name
        params = tool_call.parameters
        
        try:
            if tool_name == "load_supply_chain_data":
                return ToolRegistry._load_supply_chain_data(params, context)
            elif tool_name == "filter_product_data":
                return ToolRegistry._filter_product_data(params, context)
            elif tool_name == "calculate_demand_forecast":
                return ToolRegistry._calculate_demand_forecast(params, context)
            elif tool_name == "analyze_stockout_risk":
                return ToolRegistry._analyze_stockout_risk(params, context)
            elif tool_name == "generate_reorder_recommendations":
                return ToolRegistry._generate_reorder_recommendations(params, context)
            elif tool_name == "calculate_inventory_metrics":
                return ToolRegistry._calculate_inventory_metrics(params, context)
            elif tool_name == "compare_products":
                return ToolRegistry._compare_products(params, context)
            elif tool_name == "generate_data_table":
                return ToolRegistry._generate_data_table(params, context)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
                
        except Exception as e:
            raise Exception(f"Tool execution failed: {str(e)}")
    
    # Tool Implementation Methods
    @staticmethod
    def _load_supply_chain_data(params: Dict, context: ExecutionContext) -> Dict:
        """Load supply chain data from CSV"""
        file_path = params.get("file_path", "supply_chain_data.csv")
        
        try:
            df = pd.read_csv(file_path)
            context.data_context["raw_data"] = df
            context.data_context["data_loaded"] = True
            context.data_context["total_records"] = len(df)
            context.data_context["products"] = df['product_name'].unique().tolist()
            context.data_context["date_range"] = {
                "start": df['date'].min(),
                "end": df['date'].max()
            }
            
            return {
                "success": True,
                "records_loaded": len(df),
                "products": df['product_name'].unique().tolist(),
                "date_range": {"start": df['date'].min(), "end": df['date'].max()},
                "columns": df.columns.tolist(),
                "summary": df.describe().to_dict() if len(df) > 0 else {}
            }
        except Exception as e:
            raise Exception(f"Failed to load data: {str(e)}")
    
    @staticmethod
    def _filter_product_data(params: Dict, context: ExecutionContext) -> Dict:
        """Filter product data based on criteria"""
        if "raw_data" not in context.data_context:
            raise Exception("No data loaded. Please load data first.")
        
        df = context.data_context["raw_data"].copy()
        original_count = len(df)
        
        # Apply filters
        if "product_name" in params and params["product_name"]:
            df = df[df['product_name'] == params["product_name"]]
        
        if "product_id" in params and params["product_id"]:
            df = df[df['product_id'] == params["product_id"]]
        
        if "date_start" in params and params["date_start"]:
            df = df[df['date'] >= params["date_start"]]
        
        if "date_end" in params and params["date_end"]:
            df = df[df['date'] <= params["date_end"]]
        
        if "min_stock" in params and params["min_stock"] is not None:
            df = df[df['current_stock'] >= params["min_stock"]]
        
        if "max_stock" in params and params["max_stock"] is not None:
            df = df[df['current_stock'] <= params["max_stock"]]
        
        # Store filtered data
        context.data_context["filtered_data"] = df
        context.data_context["filter_applied"] = True
        
        return {
            "success": True,
            "original_records": original_count,
            "filtered_records": len(df),
            "filters_applied": {k: v for k, v in params.items() if v is not None},
            "resulting_products": df['product_name'].unique().tolist() if len(df) > 0 else [],
            "data_preview": df.head(5).to_dict('records') if len(df) > 0 else []
        }
    
    @staticmethod
    def _calculate_demand_forecast(params: Dict, context: ExecutionContext) -> Dict:
        """Calculate demand forecast using time series analysis"""
        product_name = params.get("product_name")
        forecast_days = params.get("forecast_days", 14)
        confidence_level = params.get("confidence_level", 0.95)
        
        # Get product data
        data_key = "filtered_data" if "filtered_data" in context.data_context else "raw_data"
        if data_key not in context.data_context:
            raise Exception("No data available for forecasting")
        
        df = context.data_context[data_key]
        
        if product_name:
            # Handle special cases for "all" products
            if product_name.lower() in ['all', 'all_products', '']:
                # Use the most recent data for all products
                product_data = df.groupby('product_name').tail(1)
            else:
                product_data = df[df['product_name'] == product_name]
                if len(product_data) == 0:
                    raise Exception(f"No data found for product: {product_name}")
        else:
            # Use the most recent data for any product
            product_data = df.groupby('product_name').tail(1)
        
        # Calculate demand forecast using simple moving average with trend
        forecasts = {}
        for _, row in product_data.iterrows():
            product = row['product_name']
            current_demand = row['daily_demand']
            
            # Simple forecasting: use recent average with some variation
            recent_data = df[df['product_name'] == product].tail(7)
            avg_demand = recent_data['daily_demand'].mean()
            
            # Generate forecast with some realistic variation
            forecast = []
            for i in range(forecast_days):
                # Add some trend and seasonality
                trend_factor = 1 + (i * 0.02)  # Slight upward trend
                seasonal_factor = 1 + 0.1 * np.sin(i * np.pi / 7)  # Weekly pattern
                random_factor = np.random.uniform(0.9, 1.1)
                
                forecast_demand = int(avg_demand * trend_factor * seasonal_factor * random_factor)
                forecast.append(max(0, forecast_demand))
            
            forecasts[product] = {
                "forecast": forecast,
                "avg_daily_demand": round(avg_demand, 2),
                "confidence_interval": {
                    "lower": [int(f * 0.8) for f in forecast],
                    "upper": [int(f * 1.2) for f in forecast]
                }
            }
        
        # Store in context
        context.analysis_results["demand_forecasts"] = forecasts
        
        return {
            "success": True,
            "forecast_days": forecast_days,
            "products_forecasted": list(forecasts.keys()),
            "forecasts": forecasts,
            "method": "Moving Average with Trend and Seasonality"
        }
    
    @staticmethod
    def _analyze_stockout_risk(params: Dict, context: ExecutionContext) -> Dict:
        """Analyze stockout risk for products"""
        product_name = params.get("product_name")
        
        # Get data
        data_key = "filtered_data" if "filtered_data" in context.data_context else "raw_data"
        if data_key not in context.data_context:
            raise Exception("No data available for risk analysis")
        
        df = context.data_context[data_key]
        
        # Get latest data for each product
        latest_data = df.groupby('product_name').tail(1)
        
        risk_analysis = {}
        for _, row in latest_data.iterrows():
            product = row['product_name']
            if product_name and product != product_name:
                continue
                
            current_stock = row['current_stock']
            daily_demand = row['daily_demand']
            lead_time = row['lead_time_days']
            
            # Calculate stockout risk
            demand_during_lead_time = daily_demand * lead_time
            stockout_probability = max(0, min(1, (demand_during_lead_time - current_stock) / demand_during_lead_time)) if demand_during_lead_time > 0 else 0
            
            # Calculate days until stockout
            days_until_stockout = int(current_stock / daily_demand) if daily_demand > 0 else float('inf')
            
            # Risk level classification
            if stockout_probability < 0.1:
                risk_level = "Low"
            elif stockout_probability < 0.3:
                risk_level = "Medium"
            else:
                risk_level = "High"
            
            risk_analysis[product] = {
                "current_stock": current_stock,
                "daily_demand": daily_demand,
                "lead_time_days": lead_time,
                "stockout_probability": round(stockout_probability, 3),
                "days_until_stockout": days_until_stockout,
                "risk_level": risk_level,
                "critical_date": (datetime.now() + timedelta(days=days_until_stockout)).strftime('%Y-%m-%d') if days_until_stockout != float('inf') else "N/A"
            }
        
        # Store in context
        context.analysis_results["stockout_risk"] = risk_analysis
        
        return {
            "success": True,
            "products_analyzed": list(risk_analysis.keys()),
            "risk_analysis": risk_analysis,
            "high_risk_products": [p for p, data in risk_analysis.items() if data["risk_level"] == "High"]
        }
    
    @staticmethod
    def _generate_reorder_recommendations(params: Dict, context: ExecutionContext) -> Dict:
        """Generate reorder recommendations based on current stock and demand"""
        product_name = params.get("product_name")
        safety_stock_ratio = params.get("safety_stock_ratio", 0.2)
        
        # Get risk analysis if available, otherwise calculate
        if "stockout_risk" not in context.analysis_results:
            # Run risk analysis first
            ToolRegistry._analyze_stockout_risk({"product_name": product_name}, context)
        
        risk_data = context.analysis_results["stockout_risk"]
        
        # Get forecast data if available
        forecast_data = None
        if "demand_forecasts" in context.analysis_results:
            forecast_data = context.analysis_results["demand_forecasts"]
        
        recommendations = {}
        for product, risk_info in risk_data.items():
            if product_name and product != product_name:
                continue
            
            current_stock = risk_info["current_stock"]
            daily_demand = risk_info["daily_demand"]
            lead_time = risk_info["lead_time_days"]
            
            # Calculate reorder point and quantity
            safety_stock = int(daily_demand * lead_time * safety_stock_ratio)
            reorder_point = int(daily_demand * lead_time) + safety_stock
            
            # Determine if reorder is needed
            reorder_needed = current_stock <= reorder_point
            
            # Calculate recommended order quantity
            if forecast_data and product in forecast_data:
                forecast = forecast_data[product]["forecast"]
                recommended_quantity = int(sum(forecast[:lead_time]) + safety_stock - current_stock)
            else:
                # Use simple calculation
                recommended_quantity = int(daily_demand * lead_time * 1.5) - current_stock
            
            recommended_quantity = max(0, recommended_quantity)
            
            # Calculate urgency
            if risk_info["days_until_stockout"] <= 7:
                urgency = "Critical"
            elif risk_info["days_until_stockout"] <= 14:
                urgency = "High"
            elif risk_info["days_until_stockout"] <= 30:
                urgency = "Medium"
            else:
                urgency = "Low"
            
            recommendations[product] = {
                "reorder_needed": reorder_needed,
                "reorder_point": reorder_point,
                "recommended_quantity": recommended_quantity,
                "urgency": urgency,
                "deadline": (datetime.now() + timedelta(days=risk_info["days_until_stockout"])).strftime('%Y-%m-%d'),
                "current_stock": current_stock,
                "safety_stock": safety_stock
            }
        
        # Store in context
        context.analysis_results["reorder_recommendations"] = recommendations
        
        return {
            "success": True,
            "products_analyzed": list(recommendations.keys()),
            "recommendations": recommendations,
            "critical_reorders": [p for p, data in recommendations.items() if data["urgency"] == "Critical" and data["reorder_needed"]]
        }
    
    @staticmethod
    def _calculate_inventory_metrics(params: Dict, context: ExecutionContext) -> Dict:
        """Calculate comprehensive inventory metrics"""
        product_name = params.get("product_name")
        time_period_days = params.get("time_period_days", 30)
        
        # Get data
        data_key = "filtered_data" if "filtered_data" in context.data_context else "raw_data"
        if data_key not in context.data_context:
            raise Exception("No data available for metrics calculation")
        
        df = context.data_context[data_key]
        
        # Filter by product if specified
        if product_name:
            df = df[df['product_name'] == product_name]
        
        metrics = {}
        for product in df['product_name'].unique():
            product_data = df[df['product_name'] == product]
            
            # Calculate metrics
            avg_stock = product_data['current_stock'].mean()
            avg_demand = product_data['daily_demand'].mean()
            stock_volatility = product_data['current_stock'].std()
            demand_volatility = product_data['daily_demand'].std()
            
            # Inventory turnover (simplified)
            total_demand = product_data['daily_demand'].sum()
            avg_inventory = product_data['current_stock'].mean()
            turnover_rate = total_demand / avg_inventory if avg_inventory > 0 else 0
            
            # Service level (simplified)
            stockout_days = len(product_data[product_data['current_stock'] == 0])
            total_days = len(product_data)
            service_level = (total_days - stockout_days) / total_days if total_days > 0 else 0
            
            metrics[product] = {
                "avg_stock": round(avg_stock, 2),
                "avg_daily_demand": round(avg_demand, 2),
                "stock_volatility": round(stock_volatility, 2),
                "demand_volatility": round(demand_volatility, 2),
                "turnover_rate": round(turnover_rate, 2),
                "service_level": round(service_level, 3),
                "total_demand": int(total_demand),
                "stockout_days": stockout_days,
                "total_days": total_days
            }
        
        # Store in context
        context.analysis_results["inventory_metrics"] = metrics
        
        return {
            "success": True,
            "products_analyzed": list(metrics.keys()),
            "metrics": metrics,
            "period_days": time_period_days
        }
    
    @staticmethod
    def _compare_products(params: Dict, context: ExecutionContext) -> Dict:
        """Compare multiple products across various metrics"""
        product_names = params.get("product_names", [])
        metrics = params.get("metrics", ["stockout_risk", "demand_forecast", "inventory_metrics"])
        
        # Get data
        data_key = "filtered_data" if "filtered_data" in context.data_context else "raw_data"
        if data_key not in context.data_context:
            raise Exception("No data available for comparison")
        
        df = context.data_context[data_key]
        
        # If no products specified, use all available
        if not product_names:
            product_names = df['product_name'].unique().tolist()
        
        comparison_data = {}
        for product in product_names:
            if product not in df['product_name'].values:
                continue
                
            product_data = df[df['product_name'] == product].iloc[-1]  # Latest data
            
            comparison_data[product] = {
                "current_stock": product_data['current_stock'],
                "daily_demand": product_data['daily_demand'],
                "lead_time_days": product_data['lead_time_days'],
                "unit_cost": product_data['unit_cost'],
                "supplier": product_data['supplier']
            }
            
            # Add calculated metrics if available
            if "stockout_risk" in context.analysis_results and product in context.analysis_results["stockout_risk"]:
                comparison_data[product]["stockout_risk"] = context.analysis_results["stockout_risk"][product]
            
            if "demand_forecasts" in context.analysis_results and product in context.analysis_results["demand_forecasts"]:
                comparison_data[product]["demand_forecast"] = context.analysis_results["demand_forecasts"][product]
            
            if "inventory_metrics" in context.analysis_results and product in context.analysis_results["inventory_metrics"]:
                comparison_data[product]["inventory_metrics"] = context.analysis_results["inventory_metrics"][product]
        
        # Store in context
        context.analysis_results["product_comparison"] = comparison_data
        
        return {
            "success": True,
            "products_compared": list(comparison_data.keys()),
            "comparison_data": comparison_data,
            "metrics_compared": metrics
        }
    
    @staticmethod
    def _generate_data_table(params: Dict, context: ExecutionContext) -> Dict:
        """Generate beautiful markdown tables from analysis results"""
        table_type = params.get("table_type", "summary")
        
        if table_type == "inventory_summary":
            return ToolRegistry._generate_inventory_summary_table(context, include_charts=True)
        
        # Generate table based on available data
        table_data = []
        
        if "stockout_risk" in context.analysis_results:
            for product, risk_data in context.analysis_results["stockout_risk"].items():
                table_data.append({
                    "Product": product,
                    "Current Stock": risk_data["current_stock"],
                    "Daily Demand": risk_data["daily_demand"],
                    "Stockout Risk": f"{risk_data['stockout_probability']:.1%}",
                    "Risk Level": risk_data["risk_level"],
                    "Days Until Stockout": risk_data["days_until_stockout"]
                })
        
        if "reorder_recommendations" in context.analysis_results:
            for product, rec_data in context.analysis_results["reorder_recommendations"].items():
                table_data.append({
                    "Product": product,
                    "Reorder Needed": "Yes" if rec_data["reorder_needed"] else "No",
                    "Recommended Qty": rec_data["recommended_quantity"],
                    "Urgency": rec_data["urgency"],
                    "Deadline": rec_data["deadline"]
                })
        
        markdown_table = ToolRegistry._convert_to_markdown_table(table_data)
        
        return {
            "success": True,
            "table_type": table_type,
            "row_count": len(table_data),
            "markdown_table": markdown_table,
            "data": table_data
        }
    
    @staticmethod
    def _generate_inventory_summary_table(context: ExecutionContext, include_charts: bool) -> Dict:
        """Generate inventory summary table"""
        data_key = "filtered_data" if "filtered_data" in context.data_context else "raw_data"
        if data_key not in context.data_context:
            raise Exception("No data available for inventory summary")
        
        df = context.data_context[data_key]
        
        # Create summary by product
        summary = df.groupby('product_name').agg({
            'current_stock': 'last',
            'daily_demand': 'mean',
            'lead_time_days': 'last',
            'unit_cost': 'last'
        }).round(2)
        
        # Convert to markdown table format
        table_data = []
        for product, data in summary.iterrows():
            table_data.append({
                "Product": product,
                "Current Stock": f"{data['current_stock']} units",
                "Avg Daily Demand": f"{data['daily_demand']:.1f} units",
                "Lead Time": f"{data['lead_time_days']} days",
                "Unit Cost": f"${data['unit_cost']:.2f}"
            })
        
        return {
            "success": True,
            "table_type": "inventory_summary",
            "table_data": table_data,
            "markdown_table": ToolRegistry._convert_to_markdown_table(table_data),
            "chart_data": summary.to_dict() if include_charts else None
        }
    
    @staticmethod
    def _convert_to_markdown_table(data: List[Dict]) -> str:
        """Convert data to markdown table format"""
        if not data:
            return ""
        
        headers = list(data[0].keys())
        
        # Create markdown table
        table = "| " + " | ".join(headers) + " |\n"
        table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        
        for row in data:
            table += "| " + " | ".join(str(row[header]) for header in headers) + " |\n"
        
        return table

# =====================================================================
# Main Tool Calling Orchestrator
# =====================================================================

class ToolCallOrchestrator:
    """Main orchestrator for tool calling with recursive processing"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.tool_registry = ToolRegistry()
        
    def process_query_with_streaming(self, user_query: str, on_tool_start=None, on_tool_complete=None, max_iterations: int = 5) -> ExecutionContext:
        """Main orchestration method with streaming callbacks"""
        return self._process_query_internal(user_query, max_iterations, on_tool_start, on_tool_complete)
    
    def process_query(self, user_query: str, max_iterations: int = 5) -> ExecutionContext:
        """Main orchestration method with recursive tool calling"""
        return self._process_query_internal(user_query, max_iterations)
    
    def _process_query_internal(self, user_query: str, max_iterations: int = 5, on_tool_start=None, on_tool_complete=None, context: Optional[ExecutionContext] = None) -> ExecutionContext:
        """Internal orchestration method with optional streaming callbacks"""
        
        # Initialize execution context if not provided
        if context is None:
            context = ExecutionContext(
                session_id=self.session_id,
                user_query=user_query,
                steps=[],
                data_context={},
                analysis_results={},
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        else:
            # Update the existing context with the new query
            context.user_query = user_query
            context.updated_at = datetime.now()
        
        print(f"🚀 Starting orchestration for query: {user_query}")
        
        iteration = 0
        while iteration < max_iterations and not context.is_complete:
            iteration += 1
            print(f"\n📋 Iteration {iteration}/{max_iterations}")
            
            # Determine what tools to call next
            next_actions = self._determine_next_actions(context, iteration)
            
            if not next_actions["tool_calls"]:
                # No more tools needed, mark as complete
                context.is_complete = True
                break
            
            # Execute the tools with streaming callbacks
            step = self._execute_tool_step_with_streaming(next_actions, context, iteration, on_tool_start, on_tool_complete)
            context.steps.append(step)
            context.updated_at = datetime.now()
            
            print(f"✅ Step {iteration} completed: {step.description}")
        
        if iteration >= max_iterations:
            print(f"⚠️  Reached maximum iterations ({max_iterations})")
        
        context.is_complete = True
        context.updated_at = datetime.now()
        print(f"🎉 Orchestration completed in {iteration} iterations")
        
        return context
    
    def _determine_next_actions(self, context: ExecutionContext, iteration: int) -> Dict:
        """Use LLM to determine what tools to call next"""
        
        # Build the context summary for the LLM
        context_summary = self._build_context_summary(context)
        
        system_prompt = f"""You are an AI orchestrator for supply chain analysis. Based on the user query and current execution context, determine what tools should be called next.

            Available tools:
            {json.dumps([tool["function"] for tool in ToolRegistry.get_tool_definitions()], indent=2)}

            Current context:
            - User Query: {context.user_query}
            - Iteration: {iteration}
            - Data Available: {bool(context.data_context)}
            - Steps Completed: {len(context.steps)}
            - Analysis Results: {list(context.analysis_results.keys())}

            Context Summary:
            {context_summary}

            Rules:
            1. Always load data first if not already loaded
            2. Filter data if specific products are mentioned
            3. For forecasting questions, call calculate_demand_forecast
            4. For risk questions, call analyze_stockout_risk
            5. For reorder questions, call generate_reorder_recommendations
            6. For comparison questions, call compare_products
            7. Always end with generate_data_table for structured output
            8. If analysis is complete, return empty tool_calls array

            Respond with the next tool calls needed, or empty array if analysis is complete."""

        try:
            print(f"🤖 Making OpenAI API call to determine next actions...")
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"What tools should be called next for this query: {context.user_query}"}
                ],
                tools=ToolRegistry.get_tool_definitions(),
                tool_choice="auto",
                temperature=0.1
            )
            
            print(f"📡 OpenAI API response received")
            
            # Extract tool calls from response
            tool_calls = []
            if response.choices[0].message.tool_calls:
                print(f"🔧 Found {len(response.choices[0].message.tool_calls)} tool calls")
                for tool_call in response.choices[0].message.tool_calls:
                    try:
                        parsed_args = json.loads(tool_call.function.arguments)
                        tool_calls.append({
                            "id": tool_call.id,
                            "name": tool_call.function.name,
                            "parameters": parsed_args
                        })
                        print(f"  📋 Tool: {tool_call.function.name} with args: {parsed_args}")
                    except json.JSONDecodeError as je:
                        print(f"❌ JSON decode error for tool {tool_call.function.name}: {je}")
                        continue
            else:
                print("⚠️ No tool calls found in OpenAI response")
                # Force basic tool calls based on query type
                query_lower = context.user_query.lower()
                tool_calls = []
                
                # Always start with data loading if not already loaded
                if not context.data_context.get("data_loaded", False):
                    tool_calls.append({
                        "id": f"forced_{uuid.uuid4().hex[:8]}",
                        "name": "load_supply_chain_data",
                        "parameters": {"file_path": "supply_chain_data.csv"}
                    })
                
                # Add analysis tools based on query content
                if "stockout" in query_lower or "risk" in query_lower:
                    tool_calls.append({
                        "id": f"forced_{uuid.uuid4().hex[:8]}",
                        "name": "analyze_stockout_risk",
                        "parameters": {}
                    })
                elif "forecast" in query_lower or "demand" in query_lower:
                    tool_calls.append({
                        "id": f"forced_{uuid.uuid4().hex[:8]}",
                        "name": "calculate_demand_forecast",
                        "parameters": {"product_name": "all", "forecast_days": 30}
                    })
                elif "reorder" in query_lower or "order" in query_lower:
                    tool_calls.append({
                        "id": f"forced_{uuid.uuid4().hex[:8]}",
                        "name": "generate_reorder_recommendations",
                        "parameters": {}
                    })
                elif "compare" in query_lower or "performance" in query_lower:
                    tool_calls.append({
                        "id": f"forced_{uuid.uuid4().hex[:8]}",
                        "name": "compare_products",
                        "parameters": {"product_names": ["all"]}
                    })
                else:
                    # Default to stockout risk analysis
                    tool_calls.append({
                        "id": f"forced_{uuid.uuid4().hex[:8]}",
                        "name": "analyze_stockout_risk",
                        "parameters": {}
                    })
                
                # Always end with data table generation
                tool_calls.append({
                    "id": f"forced_{uuid.uuid4().hex[:8]}",
                    "name": "generate_data_table",
                    "parameters": {"data_type": "inventory_summary"}
                })
                
                print(f"🔧 Force-added {len(tool_calls)} tool calls for analysis")
            
            reasoning = response.choices[0].message.content or "Determined based on query analysis"
            print(f"💭 Reasoning: {reasoning}")
            
            return {
                "tool_calls": tool_calls,
                "reasoning": reasoning
            }
            
        except Exception as e:
            print(f"❌ Error in determining next actions: {str(e)}")
            print(f"🔍 Full error details: {type(e).__name__}: {str(e)}")
            
            # Fallback tool selection based on query keywords
            fallback_tools = []
            query_lower = context.user_query.lower()
            
            # Always start with data loading
            if not context.data_context.get("data_loaded", False):
                fallback_tools.append({
                    "id": f"fallback_{uuid.uuid4().hex[:8]}",
                    "name": "load_supply_chain_data",
                    "parameters": {"file_path": "supply_chain_data.csv"}
                })
            
            # Add analysis tools based on query content
            if "stockout" in query_lower or "risk" in query_lower:
                fallback_tools.append({
                    "id": f"fallback_{uuid.uuid4().hex[:8]}",
                    "name": "analyze_stockout_risk",
                    "parameters": {}
                })
            elif "forecast" in query_lower or "demand" in query_lower:
                fallback_tools.append({
                    "id": f"fallback_{uuid.uuid4().hex[:8]}",
                    "name": "calculate_demand_forecast",
                    "parameters": {"product_name": "all", "forecast_days": 30}
                })
            elif "reorder" in query_lower or "order" in query_lower:
                fallback_tools.append({
                    "id": f"fallback_{uuid.uuid4().hex[:8]}",
                    "name": "generate_reorder_recommendations",
                    "parameters": {}
                })
            elif "compare" in query_lower or "performance" in query_lower:
                fallback_tools.append({
                    "id": f"fallback_{uuid.uuid4().hex[:8]}",
                    "name": "compare_products",
                    "parameters": {"product_names": ["all"]}
                })
            else:
                # Default to stockout risk analysis
                fallback_tools.append({
                    "id": f"fallback_{uuid.uuid4().hex[:8]}",
                    "name": "analyze_stockout_risk",
                    "parameters": {}
                })
            
            # Always end with data table generation
            fallback_tools.append({
                "id": f"fallback_{uuid.uuid4().hex[:8]}",
                "name": "generate_data_table",
                "parameters": {"data_type": "inventory_summary"}
            })
            
            print(f"🔧 Using fallback tools: {[t['name'] for t in fallback_tools]}")
            
            return {
                "tool_calls": fallback_tools,
                "reasoning": f"Using fallback tool selection due to API error: {str(e)}"
            }
    
    def _build_context_summary(self, context: ExecutionContext) -> str:
        """Build a summary of the current execution context"""
        summary = []
        
        if context.data_context:
            summary.append(f"Data Context: {list(context.data_context.keys())}")
        
        if context.analysis_results:
            summary.append(f"Analysis Results: {list(context.analysis_results.keys())}")
        
        for step in context.steps:
            summary.append(f"Step: {step.description} ({len(step.tool_calls)} tools)")
        
        return "\n".join(summary) if summary else "No previous context"
    
    def _execute_tool_step(self, next_actions: Dict, context: ExecutionContext, iteration: int) -> ExecutionStep:
        """Execute a step containing multiple tool calls"""
        
        step_id = f"step_{iteration}_{uuid.uuid4().hex[:8]}"
        step = ExecutionStep(
            step_id=step_id,
            description=f"Iteration {iteration}: {next_actions['reasoning']}",
            tool_calls=[],
            reasoning=next_actions["reasoning"],
            created_at=datetime.now()
        )
        
        # Execute each tool call
        for tool_call_def in next_actions["tool_calls"]:
            tool_call = ToolCall(
                id=tool_call_def["id"],
                name=tool_call_def["name"],
                parameters=tool_call_def["parameters"],
                status=ToolCallStatus.PENDING,
                created_at=datetime.now()
            )
            
            try:
                print(f"  🔧 Executing tool: {tool_call.name}")
                tool_call.status = ToolCallStatus.RUNNING
                tool_call.started_at = datetime.now()
                
                # Execute the tool
                result = ToolRegistry.execute_tool(tool_call, context)
                
                tool_call.result = result
                tool_call.status = ToolCallStatus.COMPLETED
                tool_call.completed_at = datetime.now()
                tool_call.execution_time = (tool_call.completed_at - tool_call.started_at).total_seconds()
                
                print(f"    ✅ {tool_call.name} completed in {tool_call.execution_time:.2f}s")
                
            except Exception as e:
                tool_call.status = ToolCallStatus.FAILED
                tool_call.error = str(e)
                tool_call.completed_at = datetime.now()
                tool_call.execution_time = (tool_call.completed_at - tool_call.started_at).total_seconds()
                
                print(f"    ❌ {tool_call.name} failed: {str(e)}")
            
            step.tool_calls.append(tool_call)
        
        step.completed_at = datetime.now()
        return step
    
    def _execute_tool_step_with_streaming(self, next_actions: Dict, context: ExecutionContext, iteration: int, on_tool_start=None, on_tool_complete=None) -> ExecutionStep:
        """Execute a step containing multiple tool calls with streaming callbacks"""
        
        step_id = f"step_{iteration}_{uuid.uuid4().hex[:8]}"
        step = ExecutionStep(
            step_id=step_id,
            description=f"Iteration {iteration}: {next_actions['reasoning']}",
            tool_calls=[],
            reasoning=next_actions["reasoning"],
            created_at=datetime.now()
        )
        
        # Execute each tool call with streaming
        for tool_call_def in next_actions["tool_calls"]:
            tool_call = ToolCall(
                id=tool_call_def["id"],
                name=tool_call_def["name"],
                parameters=tool_call_def["parameters"],
                status=ToolCallStatus.PENDING,
                created_at=datetime.now()
            )
            
            try:
                print(f"  🔧 Executing tool: {tool_call.name}")
                
                # Streaming callback for tool start
                if on_tool_start:
                    description = self._get_tool_description(tool_call.name, tool_call.parameters)
                    on_tool_start(tool_call.name, description)
                
                tool_call.status = ToolCallStatus.RUNNING
                tool_call.started_at = datetime.now()
                
                # Execute the tool
                result = ToolRegistry.execute_tool(tool_call, context)
                
                tool_call.result = result
                tool_call.status = ToolCallStatus.COMPLETED
                tool_call.completed_at = datetime.now()
                tool_call.execution_time = (tool_call.completed_at - tool_call.started_at).total_seconds()
                
                print(f"    ✅ {tool_call.name} completed in {tool_call.execution_time:.2f}s")
                
                # Streaming callback for tool completion
                if on_tool_complete:
                    summary = self._get_tool_result_summary(tool_call.name, result)
                    on_tool_complete(tool_call.name, summary)
                
            except Exception as e:
                tool_call.status = ToolCallStatus.FAILED
                tool_call.error = str(e)
                tool_call.completed_at = datetime.now()
                tool_call.execution_time = (tool_call.completed_at - tool_call.started_at).total_seconds()
                
                print(f"    ❌ {tool_call.name} failed: {str(e)}")
            
            step.tool_calls.append(tool_call)
        
        step.completed_at = datetime.now()
        return step
    
    def _get_tool_description(self, tool_name: str, parameters: Dict) -> str:
        """Get user-friendly description of what a tool does"""
        descriptions = {
            "load_supply_chain_data": "Loading supply chain data from CSV file...",
            "filter_product_data": f"Filtering data for: {parameters.get('product_name', 'specified criteria')}",
            "calculate_demand_forecast": f"Generating {parameters.get('forecast_days', 14)}-day demand forecast using AI models",
            "analyze_stockout_risk": "Analyzing stockout probabilities and risk factors",
            "generate_reorder_recommendations": "Calculating optimal reorder points and quantities",
            "calculate_inventory_metrics": "Computing inventory turnover and performance metrics",
            "compare_products": "Running comparative analysis across multiple products",
            "generate_data_table": "Creating formatted data visualization tables"
        }
        return descriptions.get(tool_name, f"Executing {tool_name.replace('_', ' ').title()}")
    
    def _get_tool_result_summary(self, tool_name: str, result: Dict) -> str:
        """Get user-friendly summary of tool results"""
        if not result or not result.get("success"):
            return "Processing completed with errors"
        
        summaries = {
            "load_supply_chain_data": f"Loaded {result.get('records_loaded', 0)} records across {len(result.get('products', []))} products",
            "filter_product_data": f"Filtered to {result.get('filtered_records', 0)} relevant records",
            "calculate_demand_forecast": f"Generated forecast with {result.get('confidence_level', 95)}% confidence",
            "analyze_stockout_risk": f"Risk analysis completed for {len(result.get('risk_by_product', {}))} products",
            "generate_reorder_recommendations": f"Generated recommendations for {len(result.get('recommendations', []))} items",
            "calculate_inventory_metrics": "Inventory performance metrics calculated",
            "compare_products": f"Comparison completed across {len(result.get('comparison_data', []))} products",
            "generate_data_table": f"Generated {result.get('table_type', 'data')} table with {result.get('row_count', 0)} rows"
        }
        return summaries.get(tool_name, "Processing completed successfully")

# =====================================================================
# Response Synthesizer - Generate ChatGPT-like Markdown Responses
# =====================================================================

class ResponseSynthesizer:
    """Synthesize final responses in beautiful markdown format"""
    
    @staticmethod
    def synthesize_response(context: ExecutionContext) -> str:
        """Generate final markdown response based on execution context"""
        
        # Analyze what was accomplished
        analysis_summary = ResponseSynthesizer._analyze_results(context)
        
        # Build the prompt for response generation
        system_prompt = """You are an expert supply chain analyst assistant. Generate a comprehensive, well-formatted markdown response based on the analysis results.

            Requirements:
            1. Use clear markdown formatting with headers, tables, bullet points
            2. Present data in beautiful tables where appropriate
            3. Include executive summary and key insights
            4. Provide actionable recommendations
            5. Use professional yet accessible language
            6. Include relevant charts/data descriptions
            7. Structure like a professional consulting report
            8. Directly answer the user's original question
            9. Use the actual analysis results to provide specific insights

            Format Guidelines:
            - Use # for main title, ## for sections, ### for subsections
            - Use tables for structured data
            - Use bullet points for recommendations
            - Use **bold** for emphasis, *italic* for notes
            - Use > blockquotes for key insights
            - Use code blocks ``` for data/calculations
            """

        try:
            # Create a comprehensive analysis summary for the LLM
            analysis_data = {
                "user_query": context.user_query,
                "data_summary": {
                    "products_available": len(context.data_context.get("products", [])),
                    "total_records": context.data_context.get("total_records", 0),
                    "date_range": context.data_context.get("date_range", {})
                },
                "analysis_results": context.analysis_results,
                "execution_summary": {
                    "total_steps": len(context.steps),
                    "tools_executed": sum(len(step.tool_calls) for step in context.steps),
                    "successful_tools": sum(1 for step in context.steps for tool in step.tool_calls if tool.status.value == "completed")
                }
            }
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"""Generate a comprehensive supply chain analysis report for this query: {context.user_query}

                    Analysis Results:
                    {json.dumps(analysis_data, indent=2, default=str)}

                    Please create a professional, well-formatted markdown response that:
                    1. Directly answers the user's question
                    2. Provides specific insights from the analysis
                    3. Includes actionable recommendations
                    4. Uses tables to present key data
                    5. Has a clear executive summary

                    Focus on being practical and actionable rather than just descriptive."""}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            return response.choices[0].message.content or ResponseSynthesizer._generate_fallback_response(context)
            
        except Exception as e:
            print(f"❌ Error synthesizing response: {str(e)}")
            return ResponseSynthesizer._generate_fallback_response(context)
    
    @staticmethod
    def _analyze_results(context: ExecutionContext) -> Dict:
        """Analyze the execution context to extract key results"""
        
        summary = {
            "query": context.user_query,
            "execution_summary": {
                "total_steps": len(context.steps),
                "total_tools_executed": sum(len(step.tool_calls) for step in context.steps),
                "successful_tools": sum(1 for step in context.steps for tool in step.tool_calls if tool.status == ToolCallStatus.COMPLETED),
                "failed_tools": sum(1 for step in context.steps for tool in step.tool_calls if tool.status == ToolCallStatus.FAILED)
            },
            "data_loaded": bool(context.data_context.get("data_loaded")),
            "products_analyzed": context.data_context.get("products", []),
            "analysis_results": {}
        }
        
        # Extract key analysis results
        for key, value in context.analysis_results.items():
            if isinstance(value, dict):
                summary["analysis_results"][key] = value
        
        # Extract tool execution results
        summary["tool_results"] = []
        for step in context.steps:
            for tool_call in step.tool_calls:
                if tool_call.status == ToolCallStatus.COMPLETED and tool_call.result:
                    summary["tool_results"].append({
                        "tool": tool_call.name,
                        "result": tool_call.result
                    })
        
        return summary
    
    @staticmethod
    def _generate_fallback_response(context: ExecutionContext) -> str:
        """Generate a fallback response if LLM synthesis fails"""
        
        response = f"# Supply Chain Analysis Report\n\n"
        response += f"**Query:** {context.user_query}\n\n"
        
        if context.data_context.get("data_loaded"):
            response += "## Data Summary\n\n"
            response += f"- **Products Available:** {len(context.data_context.get('products', []))}\n"
            response += f"- **Total Records:** {context.data_context.get('total_records', 'Unknown')}\n"
            response += f"- **Date Range:** {context.data_context.get('date_range', {}).get('start', 'Unknown')} to {context.data_context.get('date_range', {}).get('end', 'Unknown')}\n\n"
        
        # Add specific analysis results
        if context.analysis_results:
            response += "## Analysis Results\n\n"
            
            # Stockout Risk Analysis
            if "stockout_risk" in context.analysis_results:
                response += "### Stockout Risk Analysis\n\n"
                risk_data = context.analysis_results["stockout_risk"]
                for product, risk in risk_data.items():
                    response += f"**{product}:**\n"
                    response += f"- Current Stock: {risk['current_stock']}\n"
                    response += f"- Daily Demand: {risk['daily_demand']}\n"
                    response += f"- Stockout Risk: {risk['stockout_probability']:.1%}\n"
                    response += f"- Risk Level: {risk['risk_level']}\n"
                    response += f"- Days Until Stockout: {risk['days_until_stockout']}\n\n"
            
            # Reorder Recommendations
            if "reorder_recommendations" in context.analysis_results:
                response += "### Reorder Recommendations\n\n"
                rec_data = context.analysis_results["reorder_recommendations"]
                for product, rec in rec_data.items():
                    response += f"**{product}:**\n"
                    response += f"- Reorder Needed: {'Yes' if rec['reorder_needed'] else 'No'}\n"
                    if rec['reorder_needed']:
                        response += f"- Recommended Quantity: {rec['recommended_quantity']}\n"
                        response += f"- Urgency: {rec['urgency']}\n"
                        response += f"- Deadline: {rec['deadline']}\n"
                    response += "\n"
            
            # Demand Forecasts
            if "demand_forecasts" in context.analysis_results:
                response += "### Demand Forecasts\n\n"
                forecast_data = context.analysis_results["demand_forecasts"]
                for product, forecast in forecast_data.items():
                    response += f"**{product}:**\n"
                    response += f"- Average Daily Demand: {forecast['avg_daily_demand']}\n"
                    response += f"- 14-Day Forecast: {forecast['forecast'][:7]}... (showing first 7 days)\n\n"
        
        response += f"## Execution Summary\n\n"
        response += f"- **Total Steps:** {len(context.steps)}\n"
        response += f"- **Tools Executed:** {sum(len(step.tool_calls) for step in context.steps)}\n"
        response += f"- **Completed Successfully:** {sum(1 for step in context.steps for tool in step.tool_calls if tool.status.value == 'completed')}\n\n"
        
        # Add actionable insights
        response += "## Key Insights & Recommendations\n\n"
        
        if "stockout_risk" in context.analysis_results:
            high_risk_products = [p for p, data in context.analysis_results["stockout_risk"].items() if data["risk_level"] == "High"]
            if high_risk_products:
                response += f"⚠️ **High Risk Products:** {', '.join(high_risk_products)}\n"
                response += "- Immediate attention required for these products\n\n"
        
        if "reorder_recommendations" in context.analysis_results:
            critical_reorders = [p for p, data in context.analysis_results["reorder_recommendations"].items() if data.get("urgency") == "Critical"]
            if critical_reorders:
                response += f"🚨 **Critical Reorders:** {', '.join(critical_reorders)}\n"
                response += "- Place orders immediately to prevent stockouts\n\n"
        
        return response 