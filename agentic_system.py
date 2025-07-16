# File: agentic_system.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from openai import OpenAI
import time
from tqdm import tqdm
import os
import uuid
import redis
import json
import pickle
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Redis connection
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)

# ----------------------------
# Module 1: Data Tools
# ----------------------------
class DataTools:
    @staticmethod
    def csv_reader(file_path):
        """Load CSV data into DataFrame"""
        return pd.read_csv(file_path)
    
    @staticmethod
    def data_filter(df, product_name=None, date_range=None):
        """Filter dataset based on parameters"""
        if product_name:
            df = df[df['product_name'] == product_name]
        if date_range:
            df = df[(df['date'] >= date_range[0]) & (df['date'] <= date_range[1])]
        return df
    
    @staticmethod
    def feature_engineer(df, window=7):
        """Calculate derived features"""
        df['avg_daily_demand'] = df['daily_demand'].rolling(window).mean()
        df['stockout_risk'] = df['current_stock'] / (df['avg_daily_demand'] * df['lead_time_days'])
        return df

# ----------------------------
# Module 2: Analytics Tools
# ----------------------------
class AnalyticsTools:
    @staticmethod
    def demand_forecaster(df, horizon=14):
        """Simulated demand forecasting (replace with actual ML model)"""
        last_demand = df['daily_demand'].iloc[-1]
        return [max(0, int(last_demand * np.random.uniform(0.8, 1.2))) for _ in range(horizon)]
    
    @staticmethod
    def stockout_risk_assessor(forecast, current_stock, lead_time):
        """Calculate stockout probability"""
        demand_sum = sum(forecast[:lead_time])
        stockout_prob = max(0, min(1, (demand_sum - current_stock) / demand_sum)) if demand_sum > 0 else 0
        
        # Fix: Convert numpy.int64 to regular Python int
        cumulative_demand = np.cumsum(forecast)
        days_until_stockout = np.argmax(cumulative_demand > current_stock)
        days_until_stockout = int(days_until_stockout)  # Convert to Python int
        
        # Handle edge case where stock never runs out
        if cumulative_demand[days_until_stockout] <= current_stock:
            days_until_stockout = len(forecast)
        
        critical_day = (datetime.today() + timedelta(days=days_until_stockout)).strftime('%Y-%m-%d')
        
        return {
            "stockout_probability": round(stockout_prob, 2),
            "critical_date": critical_day,
            "days_until_stockout": days_until_stockout
        }

# ----------------------------
# Module 3: Business Logic Tools
# ----------------------------
class BusinessTools:
    @staticmethod
    def replenishment_advisor(current_stock, forecast, unit_cost, min_threshold=0.1):
        """Generate replenishment recommendations"""
        safety_stock = int(sum(forecast) * min_threshold)
        required = sum(forecast) + safety_stock - current_stock
        reorder = required > 0
        deadline = (datetime.today() + timedelta(days=7)).strftime('%Y-%m-%d')
        return {
            "reorder_recommended": reorder,
            "order_quantity": max(0, required),
            "deadline": deadline
        }

# ----------------------------
# Module 4: Advanced Deep Analysis Tools
# ----------------------------

class StockoutRiskAnalyzer:
    """Advanced stockout risk analysis with multiple risk factors"""
    
    @staticmethod
    def comprehensive_stockout_analysis(df, product_name=None):
        """
        Perform comprehensive stockout risk analysis including:
        - Probability distributions
        - Risk timeline
        - Critical periods identification
        - Multi-factor risk scoring
        """
        if product_name:
            df = df[df['product_name'] == product_name]
        
        results = {}
        
        for product in df['product_name'].unique():
            product_data = df[df['product_name'] == product].copy()
            product_data = product_data.sort_values('date')
            
            # Calculate advanced metrics
            demand_volatility = product_data['daily_demand'].std() / product_data['daily_demand'].mean()
            avg_lead_time = product_data['lead_time_days'].iloc[-1]
            current_stock = product_data['current_stock'].iloc[-1]
            avg_demand = product_data['daily_demand'].mean()
            
            # Risk factors calculation
            demand_trend = np.polyfit(range(len(product_data)), product_data['daily_demand'], 1)[0]
            stock_trend = np.polyfit(range(len(product_data)), product_data['current_stock'], 1)[0]
            
            # Advanced stockout probability using Monte Carlo simulation
            simulation_days = 30
            num_simulations = 1000
            stockout_count = 0
            
            for _ in range(num_simulations):
                sim_stock = current_stock
                for day in range(simulation_days):
                    # Simulate demand with volatility
                    sim_demand = max(0, np.random.normal(avg_demand, avg_demand * demand_volatility))
                    sim_stock -= sim_demand
                    if sim_stock <= 0:
                        stockout_count += 1
                        break
            
            stockout_probability = stockout_count / num_simulations
            
            # Calculate days until critical stock level (safety stock)
            safety_stock = avg_demand * avg_lead_time * 1.5  # 1.5x safety factor
            days_to_safety = max(0, (current_stock - safety_stock) / avg_demand) if avg_demand > 0 else float('inf')
            
            # Risk score (0-100, where 100 is highest risk)
            risk_score = min(100, (
                stockout_probability * 40 +  # 40% weight on stockout probability
                (1 - min(current_stock / (avg_demand * avg_lead_time * 2), 1)) * 30 +  # 30% on current stock ratio
                min(demand_volatility, 1) * 20 +  # 20% on demand volatility
                max(0, -stock_trend / avg_demand) * 10  # 10% on negative stock trend
            ))
            
            results[product] = {
                "stockout_probability": round(stockout_probability, 3),
                "risk_score": round(risk_score, 1),
                "days_to_safety_stock": round(days_to_safety, 1),
                "current_stock": current_stock,
                "safety_stock_recommended": round(safety_stock, 0),
                "demand_volatility": round(demand_volatility, 3),
                "demand_trend": "increasing" if demand_trend > 0 else "decreasing" if demand_trend < 0 else "stable",
                "stock_trend": "increasing" if stock_trend > 0 else "decreasing" if stock_trend < 0 else "stable",
                "risk_level": "HIGH" if risk_score > 70 else "MEDIUM" if risk_score > 40 else "LOW"
            }
        
        return results

class SupplierPerformanceAnalyzer:
    """Comprehensive supplier performance analysis"""
    
    @staticmethod
    def analyze_supplier_performance(df):
        """
        Analyze supplier performance across multiple dimensions:
        - Cost efficiency
        - Lead time reliability
        - Supply consistency
        - Risk assessment
        """
        results = {}
        
        for supplier in df['supplier'].unique():
            supplier_data = df[df['supplier'] == supplier].copy()
            
            # Cost analysis
            avg_unit_cost = supplier_data['unit_cost'].mean()
            cost_volatility = supplier_data['unit_cost'].std() / avg_unit_cost if avg_unit_cost > 0 else 0
            
            # Lead time analysis
            avg_lead_time = supplier_data['lead_time_days'].mean()
            lead_time_consistency = 1 - (supplier_data['lead_time_days'].std() / avg_lead_time) if avg_lead_time > 0 else 1
            
            # Product diversity
            products_supplied = supplier_data['product_name'].nunique()
            total_volume = len(supplier_data)
            
            # Calculate stock coverage (how well supplier maintains stock)
            stockout_days = len(supplier_data[supplier_data['current_stock'] == 0])
            stock_reliability = 1 - (stockout_days / len(supplier_data)) if len(supplier_data) > 0 else 0
            
            # Demand fulfillment capability
            avg_demand = supplier_data['daily_demand'].mean()
            avg_stock = supplier_data['current_stock'].mean()
            demand_coverage = avg_stock / (avg_demand * avg_lead_time) if (avg_demand * avg_lead_time) > 0 else 0
            
            # Overall performance score (0-100)
            performance_score = (
                min(stock_reliability, 1) * 35 +  # 35% weight on stock reliability
                min(lead_time_consistency, 1) * 25 +  # 25% on lead time consistency
                min(demand_coverage / 2, 1) * 20 +  # 20% on demand coverage
                (1 - min(cost_volatility, 1)) * 10 +  # 10% on cost stability
                min(products_supplied / 3, 1) * 10  # 10% on product diversity
            )
            
            results[supplier] = {
                "performance_score": round(performance_score, 1),
                "avg_unit_cost": round(avg_unit_cost, 2),
                "cost_volatility": round(cost_volatility, 3),
                "avg_lead_time": round(avg_lead_time, 1),
                "lead_time_consistency": round(lead_time_consistency, 3),
                "stock_reliability": round(stock_reliability, 3),
                "demand_coverage_ratio": round(demand_coverage, 2),
                "products_supplied": products_supplied,
                "total_volume": total_volume,
                "stockout_incidents": stockout_days,
                "grade": "A" if performance_score > 85 else "B" if performance_score > 70 else "C" if performance_score > 55 else "D"
            }
        
        return results

class InventoryOptimizationCalculator:
    """Advanced inventory optimization calculations"""
    
    @staticmethod
    def calculate_optimal_inventory_levels(df, carrying_cost_rate=0.25, stockout_cost_multiplier=5):
        """
        Calculate optimal inventory levels using advanced inventory theory:
        - Economic Order Quantity (EOQ)
        - Reorder Point (ROP)
        - Safety Stock optimization
        - Service level optimization
        """
        results = {}
        
        for product in df['product_name'].unique():
            product_data = df[df['product_name'] == product].copy()
            product_data = product_data.sort_values('date')
            
            # Key parameters
            avg_demand = product_data['daily_demand'].mean()
            demand_std = product_data['daily_demand'].std()
            unit_cost = product_data['unit_cost'].iloc[-1]
            lead_time = product_data['lead_time_days'].iloc[-1]
            current_stock = product_data['current_stock'].iloc[-1]
            
            # EOQ calculation (simplified - assumes ordering cost)
            annual_demand = avg_demand * 365
            ordering_cost = unit_cost * 0.1  # Assume 10% of unit cost as ordering cost
            carrying_cost = unit_cost * carrying_cost_rate
            
            eoq = np.sqrt((2 * annual_demand * ordering_cost) / carrying_cost) if carrying_cost > 0 else avg_demand * 30
            
            # Safety stock using service level approach (95% service level)
            service_level = 0.95
            z_score = 1.65  # For 95% service level
            safety_stock = z_score * demand_std * np.sqrt(lead_time) if demand_std > 0 else avg_demand * lead_time * 0.2
            
            # Reorder point
            reorder_point = (avg_demand * lead_time) + safety_stock
            
            # Maximum stock level
            max_stock = reorder_point + eoq
            
            # Current inventory position analysis
            days_of_inventory = current_stock / avg_demand if avg_demand > 0 else 0
            inventory_turnover = annual_demand / current_stock if current_stock > 0 else float('inf')
            
            # Cost analysis
            holding_cost_per_day = current_stock * unit_cost * (carrying_cost_rate / 365)
            annual_holding_cost = holding_cost_per_day * 365
            
            # Optimization recommendations
            if current_stock < safety_stock:
                action = "URGENT_REORDER"
                quantity_needed = reorder_point - current_stock
            elif current_stock < reorder_point:
                action = "REORDER_SOON"
                quantity_needed = eoq
            elif current_stock > max_stock:
                action = "EXCESS_INVENTORY"
                quantity_needed = -(current_stock - max_stock)
            else:
                action = "OPTIMAL_RANGE"
                quantity_needed = 0
                
            results[product] = {
                "current_stock": current_stock,
                "optimal_reorder_point": round(reorder_point, 0),
                "optimal_safety_stock": round(safety_stock, 0),
                "economic_order_quantity": round(eoq, 0),
                "maximum_stock_level": round(max_stock, 0),
                "days_of_inventory": round(days_of_inventory, 1),
                "inventory_turnover": round(inventory_turnover, 2),
                "annual_holding_cost": round(annual_holding_cost, 2),
                "action_required": action,
                "quantity_adjustment": round(quantity_needed, 0),
                "service_level": f"{service_level*100}%",
                "optimization_score": round(min(100, (1 / max(abs(current_stock - reorder_point) / reorder_point, 0.01)) * 100), 1)
            }
        
        return results

class FinancialImpactAnalyzer:
    """Financial impact analysis for inventory decisions"""
    
    @staticmethod
    def analyze_financial_impact(df, carrying_cost_rate=0.25, stockout_cost_multiplier=5, profit_margin=0.3):
        """
        Comprehensive financial impact analysis:
        - Carrying costs
        - Stockout costs
        - Revenue impact
        - Cash flow analysis
        - ROI optimization
        """
        results = {}
        total_analysis = {
            "total_inventory_value": 0,
            "total_carrying_cost": 0,
            "total_potential_revenue": 0,
            "total_stockout_cost": 0,
            "net_financial_impact": 0
        }
        
        for product in df['product_name'].unique():
            product_data = df[df['product_name'] == product].copy()
            product_data = product_data.sort_values('date')
            
            # Key metrics
            current_stock = product_data['current_stock'].iloc[-1]
            unit_cost = product_data['unit_cost'].iloc[-1]
            avg_demand = product_data['daily_demand'].mean()
            total_demand = product_data['daily_demand'].sum()
            
            # Calculate inventory value
            inventory_value = current_stock * unit_cost
            
            # Carrying costs (annual)
            annual_carrying_cost = inventory_value * carrying_cost_rate
            
            # Revenue calculations
            unit_selling_price = unit_cost / (1 - profit_margin)  # Reverse calculate selling price
            potential_annual_revenue = avg_demand * 365 * unit_selling_price
            
            # Stockout analysis
            stockout_days = len(product_data[product_data['current_stock'] == 0])
            lost_sales = stockout_days * avg_demand
            stockout_cost = lost_sales * unit_selling_price * stockout_cost_multiplier
            
            # Cash flow impact
            days_of_inventory = current_stock / avg_demand if avg_demand > 0 else 0
            cash_tied_up = inventory_value
            opportunity_cost = cash_tied_up * 0.05  # Assume 5% opportunity cost
            
            # Profit analysis
            gross_profit_per_unit = unit_selling_price - unit_cost
            annual_gross_profit = avg_demand * 365 * gross_profit_per_unit
            
            # Working capital efficiency
            inventory_turnover = (avg_demand * 365) / current_stock if current_stock > 0 else float('inf')
            working_capital_efficiency = min(inventory_turnover / 12, 1)  # Normalize to monthly turns
            
            # Financial health score (0-100)
            financial_score = (
                min(working_capital_efficiency, 1) * 40 +  # 40% on inventory turnover
                min(1 - (annual_carrying_cost / potential_annual_revenue), 1) * 30 +  # 30% on carrying cost efficiency
                min(1 - (stockout_cost / potential_annual_revenue), 1) * 20 +  # 20% on stockout cost impact
                min(annual_gross_profit / inventory_value, 1) * 10  # 10% on profit to inventory ratio
            ) * 100
            
            # ROI calculation
            annual_roi = (annual_gross_profit - annual_carrying_cost - stockout_cost) / inventory_value if inventory_value > 0 else 0
            
            results[product] = {
                "inventory_value": round(inventory_value, 2),
                "annual_carrying_cost": round(annual_carrying_cost, 2),
                "potential_annual_revenue": round(potential_annual_revenue, 2),
                "stockout_cost": round(stockout_cost, 2),
                "cash_tied_up": round(cash_tied_up, 2),
                "opportunity_cost": round(opportunity_cost, 2),
                "annual_gross_profit": round(annual_gross_profit, 2),
                "inventory_turnover": round(inventory_turnover, 2),
                "annual_roi": f"{round(annual_roi * 100, 1)}%",
                "financial_health_score": round(financial_score, 1),
                "days_of_inventory": round(days_of_inventory, 1),
                "unit_selling_price": round(unit_selling_price, 2),
                "profit_margin_actual": f"{round(profit_margin * 100, 1)}%",
                "lost_sales_units": round(lost_sales, 0),
                "working_capital_efficiency": round(working_capital_efficiency, 3)
            }
            
            # Add to totals
            total_analysis["total_inventory_value"] += inventory_value
            total_analysis["total_carrying_cost"] += annual_carrying_cost
            total_analysis["total_potential_revenue"] += potential_annual_revenue
            total_analysis["total_stockout_cost"] += stockout_cost
        
        # Calculate net financial impact
        total_analysis["net_financial_impact"] = (
            total_analysis["total_potential_revenue"] - 
            total_analysis["total_carrying_cost"] - 
            total_analysis["total_stockout_cost"]
        )
        
        results["portfolio_summary"] = {
            "total_inventory_value": round(total_analysis["total_inventory_value"], 2),
            "total_annual_carrying_cost": round(total_analysis["total_carrying_cost"], 2),
            "total_potential_revenue": round(total_analysis["total_potential_revenue"], 2),
            "total_stockout_cost": round(total_analysis["total_stockout_cost"], 2),
            "net_financial_impact": round(total_analysis["net_financial_impact"], 2),
            "portfolio_roi": f"{round((total_analysis['net_financial_impact'] / total_analysis['total_inventory_value']) * 100, 1)}%" if total_analysis['total_inventory_value'] > 0 else "N/A"
        }
        
        return results

# ----------------------------
# Module 5: Redis-Based Session Manager
# ----------------------------
class SessionManager:
    def __init__(self, session_id):
        """Initialize with client-provided session ID"""
        if not session_id:
            raise ValueError("Session ID must be provided by client")
        
        self.session_id = session_id
        self.redis_key = f"session:{session_id}"
        
        # Check if session exists in Redis
        if not redis_client.exists(self.redis_key):
            # Create new session
            session_data = {
                "session_id": session_id,
                "session_start": datetime.now().isoformat(),
                "query_count": 0,
                "total_processing_time": 0.0,
                "message_history": [],
                "last_activity": datetime.now().isoformat(),
                "current_analysis": {}
            }
            redis_client.set(self.redis_key, pickle.dumps(session_data))
            redis_client.expire(self.redis_key, 1800)  # 30 minutes TTL
        
        self._update_last_activity()
    
    def _get_session_data(self):
        """Get session data from Redis"""
        data = redis_client.get(self.redis_key)
        if data:
            return pickle.loads(data)
        return None
    
    def _save_session_data(self, data):
        """Save session data to Redis"""
        redis_client.set(self.redis_key, pickle.dumps(data))
        redis_client.expire(self.redis_key, 1800)  # Reset TTL
    
    def _update_last_activity(self):
        """Update last activity timestamp"""
        data = self._get_session_data()
        if data:
            data["last_activity"] = datetime.now().isoformat()
            self._save_session_data(data)
        
    def add_message_to_history(self, message):
        """Add message to history in OpenAI format"""
        data = self._get_session_data()
        if data:
            data["message_history"].append(message)
            data["last_activity"] = datetime.now().isoformat()
            self._save_session_data(data)
        
    def get_message_history(self):
        """Get message history in OpenAI format"""
        data = self._get_session_data()
        return data["message_history"] if data else []
        
    # def get_conversation_history(self):
    #     """Get conversation history (backward compatibility)"""
    #     message_history = self.get_message_history()
    #     conversations = []
        
    #     for i in range(0, len(message_history), 2):
    #         if i + 1 < len(message_history):
    #             user_msg = message_history[i]
    #             assistant_msg = message_history[i + 1]
                
    #             # Extract text content from user message
    #             user_content = ""
    #             if isinstance(user_msg.get('content'), list):
    #                 for content_item in user_msg['content']:
    #                     if isinstance(content_item, dict) and 'text' in content_item:
    #                         user_content = content_item['text']
    #                         break
    #             else:
    #                 user_content = user_msg.get('content', '')
                
    #             conversations.append((user_content, assistant_msg.get('content', '')))
        
    #     return conversations
        
    def log_query_start(self, query):
        """Log the start of a new query"""
        data = self._get_session_data()
        if data:
            data["query_count"] += 1
            data["current_query_start"] = time.time()
            data["current_query"] = query
            data["last_activity"] = datetime.now().isoformat()
            self._save_session_data(data)
            print(f"[{self.session_id}] Query #{data['query_count']}: {query}", flush=True)
        
    def log_query_end(self, response):
        """Log the end of query processing"""
        data = self._get_session_data()
        if data and "current_query_start" in data:
            query_time = time.time() - data["current_query_start"]
            data["total_processing_time"] += query_time
            data["last_activity"] = datetime.now().isoformat()
            
            # Clean up temporary data
            if "current_query_start" in data:
                del data["current_query_start"]
            if "current_query" in data:
                del data["current_query"]
            
            self._save_session_data(data)
            print(f"[{self.session_id}] Query #{data['query_count']} completed in {query_time:.2f}s", flush=True)
        
    def get_session_info(self):
        """Get current session information"""
        data = self._get_session_data()
        if not data:
            return {}
        
        current_time = datetime.now()
        session_start = datetime.fromisoformat(data["session_start"])
        last_activity = datetime.fromisoformat(data["last_activity"])
        
        session_duration = current_time - session_start
        idle_time = current_time - last_activity
        
        return {
            "session_id": data["session_id"],
            "started": session_start.strftime('%Y-%m-%d %H:%M:%S'),
            "last_activity": last_activity.strftime('%Y-%m-%d %H:%M:%S'),
            "duration": str(session_duration).split('.')[0],
            "idle_time": str(idle_time).split('.')[0],
            "queries_processed": data["query_count"],
            "total_processing_time": f"{data['total_processing_time']:.2f}s",
            "avg_query_time": f"{data['total_processing_time']/max(1, data['query_count']):.2f}s",
            "message_count": len(data["message_history"]),
            "has_analysis_context": bool(data["current_analysis"])
        }
        
    def get_session_summary(self):
        """Get final session summary"""
        data = self._get_session_data()
        if not data:
            return {}
        
        session_end = datetime.now()
        session_start = datetime.fromisoformat(data["session_start"])
        session_duration = session_end - session_start
        
        return {
            "session_id": data["session_id"],
            "started": session_start.strftime('%Y-%m-%d %H:%M:%S'),
            "ended": session_end.strftime('%Y-%m-%d %H:%M:%S'),
            "total_duration": str(session_duration).split('.')[0],
            "queries_processed": data["query_count"],
            "total_processing_time": f"{data['total_processing_time']:.2f}s",
            "avg_query_time": f"{data['total_processing_time']/max(1, data['query_count']):.2f}s",
            "total_messages": len(data["message_history"])
        }
    
    def is_expired(self, timeout_minutes=30):
        """Check if session has expired based on inactivity"""
        data = self._get_session_data()
        if not data:
            return True
        
        last_activity = datetime.fromisoformat(data["last_activity"])
        idle_time = datetime.now() - last_activity
        return idle_time.total_seconds() > (timeout_minutes * 60)

# ----------------------------
# Helper Functions for Message Management
# ----------------------------
def add_message_to_history(session_id, message):
    """Add message to session history"""
    system = SessionFactory.get_session(session_id)
    if system:
        system.session.add_message_to_history(message)
        return system.session.get_message_history()
    return []

def build_complete_message_object(session_id, current_prompt, session_history):
    """Build complete message object for OpenAI API"""
    system = SessionFactory.get_session(session_id)
    
    if not system:
        print(f"No session found for session_id: {session_id}")
        return [{"role": "user", "content": current_prompt}]
    
    # Get existing message history from Redis
    messages = system.session.get_message_history().copy()
    
    # Create the current user message with simple string content
    current_user_message = {
        "role": "user", 
        "content": current_prompt
    }
    
    # Add current message to the list (but don't save to Redis yet)
    messages.append(current_user_message)
    
    # Store the user message in session history in Redis
    system.session.add_message_to_history(current_user_message)
    
    return messages

# ----------------------------
# Updated Orchestrator with New Message Structure
# ----------------------------
class Orchestrator:
    # def __init__(self, session_id, system):
    #     self.session_id = session_id
    #     self.system = system
        
    TOOL_DESCRIPTIONS = [
        {
            "name": "csv_reader",
            "description": "Load supply chain data from CSV file",
            "parameters": {"type": "object", "properties": {}}
        },
        {
            "name": "data_filter",
            "description": "Filter dataset by product and date range",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string"},
                    "date_range": {"type": "array", "items": {"type": "string"}}
                }
            }
        },
        {
            "name": "feature_engineer",
            "description": "Calculate derived metrics like stockout risk",
            "parameters": {
                "type": "object",
                "properties": {
                    "window": {"type": "integer"}
                }
            }
        },
        {
            "name": "demand_forecaster",
            "description": "Predict future demand for inventory items",
            "parameters": {
                "type": "object",
                "properties": {
                    "horizon": {"type": "integer"}
                }
            }
        }
    ]

    # @classmethod
    # def decide_tools_and_generate_insights(cls, user_query,session_history):
    #     """Use OpenAI to determine which tools to call and generate insights"""
    #     combined_system_prompt = """
    #     You are an AI assistant that helps with supply chain analysis. You perform two key functions:

    #     ## 1. TOOL SELECTION & EXECUTION
    #     Analyze the user query and determine which tool(s) to use from the available options:
    #     1. csv_reader - Load supply chain data from CSV file (use this first if data needs to be loaded)
    #     2. data_filter - Filter dataset by product name or date range
    #     3. feature_engineer - Calculate derived metrics like moving averages and stockout risk
    #     4. demand_forecaster - Predict future demand for inventory planning
        
    #     Query analysis steps:
    #     - If asking about specific products, you'll need data_filter
    #     - If asking about predictions/forecasts, you'll need demand_forecaster
    #     - If asking about risk assessment, you'll need feature_engineer
    #     - Always start with csv_reader if data isn't loaded yet

    #     ## 2. INSIGHT GENERATION & FORMATTING
    #     Provide comprehensive insights based on analysis results using clean, well-structured Markdown:
        
    #     **Formatting Guidelines:**
    #     - Use headers (# ## ###) to organize information
    #     - Use **bold** for important terms and *italic* for emphasis
    #     - Use bullet points (-) or numbered lists (1.) for clarity
    #     - Use code blocks (```) for any data, formulas, or technical content
    #     - Use tables when presenting structured data
    #     - Use blockquotes (>) for important callouts or summaries
        
    #     **Response Structure:**
    #     1. First, identify and execute the appropriate tool(s) with proper parameters
    #     2. Then, analyze the results and provide actionable insights
    #     3. Format your complete response using markdown for maximum readability
        
    #     Be concise, clear, and well-structured in your responses while ensuring both tool execution and insight delivery.
    #     """
    #     # Example implementation for tool calling (commented out):
    #     # response = client.chat.completions.create(
    #     #     model="gpt-3.5-turbo",
    #     #     messages=[{"role": "system", "content": combined_system_prompt},
    #     #               {"role": "user", "content": user_query}],
    #     #     functions=cls.TOOL_DESCRIPTIONS,
    #     #     function_call="auto"
    #     # )
    #     # return response.choices[0].message.function_call

    @classmethod
    def decide_tools_and_generate_insights(cls, user_query, session_history, session_id, system):
        """Enhanced method that determines tools, executes them recursively, and generates final response"""
        
        try:
            start_time = time.time()
            # Import required modules for real-time emissions
            from flask_socketio import emit
            from datetime import datetime
            
            # # Emit processing start
            # try:
            #     emit('processing_start', {
            #         'session_id': session_id,
            #         'message_id': f'processing-{int(time.time() * 1000)}',
            #         'query': user_query,
            #         'timestamp': datetime.now().isoformat()
            #     })
            # except Exception as e:
            #     print(f"❌ Failed to emit processing_start: {e}")
            #     import traceback
            #     print(f"🔍 [TRACEBACK] Emit processing_start error: {traceback.format_exc()}")
            
            print(f"🚀 Starting analysis for query: {user_query[:100]}...")
            
            # Get the session
            # from agentic_system import SessionFactory
            # system = SessionFactory.get_session(session_id)
            # if not system:
            #     print(f"No session found for session_id: {session_id}")
            #     return {
            #         'response': "Session not found. Please refresh and try again.",
            #         'processing_time': 0,
            #         'tools_used': [],
            #         'success': False,
            #         'error': 'Session not found'
            #     }
            
            # Import required classes
            # from orchestration_system import ExecutionContext, ResponseSynthesizer, ToolRegistry, ToolCall, ToolCallStatus
            
            # Initialize execution context
            # execution_context = ExecutionContext(
            #     session_id=session_id,
            #     user_query=user_query,
            #     steps=[],
            #     data_context={} if 'current_data' not in system.context else {'data_loaded': True, 'raw_data': system.context['current_data']},
            #     analysis_results={},
            #     created_at=datetime.now(),
            #     updated_at=datetime.now()
            # )
            
            # Define tool descriptions for OpenAI function calling
            tool_definitions = [
                {
                    "type": "function",
                    "function": {
                        "name": "csv_reader",
                        "description": "Load supply chain data from CSV file",
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
                        "name": "data_filter",
                        "description": "Filter dataset by product name or date range",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "product_name": {"type": "string", "description": "Product name to filter by"},
                                "date_range": {
                                    "type": "array", 
                                    "description": "Date range [start, end] to filter by",
                                    "items": {"type": "string"},
                                    "minItems": 2,
                                    "maxItems": 2
                                }
                            }
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "feature_engineer", 
                        "description": "Calculate derived metrics like moving averages and stockout risk",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "window": {"type": "integer", "description": "Window size for rolling calculations", "default": 7}
                            }
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "demand_forecaster",
                        "description": "Predict future demand for inventory planning",
                        "parameters": {
                            "type": "object", 
                            "properties": {
                                "horizon": {"type": "integer", "description": "Forecast horizon in days", "default": 14}
                            }
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "comprehensive_stockout_analysis",
                        "description": "Perform advanced stockout risk analysis with Monte Carlo simulation and multi-factor risk scoring",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "product_name": {
                                    "type": "string", 
                                    "description": "Specific product to analyze (optional - if not provided, analyzes all products)"
                                }
                            }
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "analyze_supplier_performance",
                        "description": "Comprehensive supplier performance analysis including cost efficiency, lead time reliability, and supply consistency",
                        "parameters": {
                            "type": "object",
                            "properties": {}
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "calculate_optimal_inventory_levels",
                        "description": "Calculate optimal inventory levels using EOQ, reorder points, and safety stock optimization",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "carrying_cost_rate": {
                                    "type": "number",
                                    "description": "Annual carrying cost rate (default: 0.25 = 25%)",
                                    "default": 0.25
                                },
                                "stockout_cost_multiplier": {
                                    "type": "number",
                                    "description": "Stockout cost multiplier (default: 5)",
                                    "default": 5
                                }
                            }
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "analyze_financial_impact",
                        "description": "Comprehensive financial impact analysis including carrying costs, stockout costs, revenue impact, and ROI optimization",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "carrying_cost_rate": {
                                    "type": "number",
                                    "description": "Annual carrying cost rate (default: 0.25 = 25%)",
                                    "default": 0.25
                                },
                                "stockout_cost_multiplier": {
                                    "type": "number",
                                    "description": "Stockout cost multiplier (default: 5)",
                                    "default": 5
                                },
                                "profit_margin": {
                                    "type": "number",
                                    "description": "Profit margin (default: 0.3 = 30%)",
                                    "default": 0.3
                                }
                            }
                        }
                    }
                }
            ]
            
            max_iterations = 1
            iteration_count = 0
            all_tool_results = []
            
            # Recursive tool execution loop
            while iteration_count < max_iterations:
                iteration_count += 1
                print(f"🔄 Tool selection iteration {iteration_count}")
                
                # # Emit iteration start
                # emit('synthesis_progress', {
                #     'session_id': session_id,
                #     'message': f"Analysis iteration {iteration_count}/{max_iterations}",
                #     'timestamp': datetime.now().isoformat()
                # })
                
                # Build context summary for LLM
                # context_summary = {
                #     "data_loaded": execution_context.data_context.get('data_loaded', False),
                #     "tools_executed": [result['tool_name'] for result in all_tool_results],
                #     "current_iteration": iteration_count,
                #     "products_available": execution_context.data_context.get('products', []),
                #     "analysis_completed": execution_context.analysis_results
                # }
                
                # print(f"📋 History -->>  \n\n {session_history}")
                
                # System message for tool selection
                system_message = f"""You are an AI assistant for supply chain analysis. Based on the user query and current context, determine what tools need to be executed.

                Available tools:
                - csv_reader: Load data from CSV (use first if no data loaded)
                - data_filter: Filter by product/date (use if specific products mentioned)
                - feature_engineer: Calculate basic risk metrics (use for simple risk analysis)  
                - demand_forecaster: Predict future demand (use for forecasting)
                - comprehensive_stockout_analysis: Advanced stockout risk analysis with Monte Carlo simulation
                - analyze_supplier_performance: Comprehensive supplier performance analysis
                - calculate_optimal_inventory_levels: Calculate EOQ, reorder points, and optimal stock levels
                - analyze_financial_impact: Financial impact analysis including ROI, carrying costs, and revenue impact
                
                User query: {user_query}
                
                Rules:
                1. If no data loaded, start with csv_reader
                2. Use data_filter for specific product queries
                3. Use feature_engineer for basic risk assessment
                4. Use demand_forecaster for demand predictions
                5. Use comprehensive_stockout_analysis for detailed risk analysis
                6. Use analyze_supplier_performance for supplier evaluation
                7. Use calculate_optimal_inventory_levels for inventory optimization
                8. Use analyze_financial_impact for financial analysis
                9. If analysis is complete and no more tools needed, don't call any function
                
                Analyze the query and context, then call the appropriate tool(s) or no tools if analysis is complete.
                
                At last once you think there is no tool is required, please follow the below rules to generate the final response
                
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
                
                Note: Provide a brief chain of thought process before calling the tool. Also if tool is picked then provide content as well."""
                
                print(f"🤖 Calling OpenAI for tool selection...")
                # print(f"📋 History -->>  \n\n {session_history}")
                
                try:
                    if "system" not in session_history:
                        session_history.insert(0, {"role": "system", "content": system_message})
                    # Call OpenAI with function calling (with timeout)
                    # print(f"📋 History -->>  \n\n {session_history}")
                    response = client.chat.completions.create(
                        model="o3-mini",
                        # messages=[
                        #     {"role": "system", "content": system_message},
                        #     {"role": "user", "content": user_query}
                        # ],
                        messages=session_history,
                        # tools=tool_definitions,  # type: ignore
                        # tool_choice="auto",
                        # temperature=0.2,
                        # timeout=30  # 30 second timeout
                    )
                    
                    print(f"✅ OpenAI response received")
                    
                except Exception as e:
                    print(f"❌ OpenAI API call failed: {str(e)}")
                    import traceback
                    print(f"🔍 [TRACEBACK] OpenAI API call error: {traceback.format_exc()}")
                    # emit('error', {
                    #     'session_id': session_id,
                    #     'error_type': 'api_error',
                    #     'message': f"OpenAI API call failed: {str(e)}",
                    #     'timestamp': datetime.now().isoformat()
                    # })
                    break
                
                # Check if tools were called
                thought_process_content = response.choices[0].message.content
                thought_process_object = {
                    "role": "assistant",
                    "content": thought_process_content
                }
                session_history.append(thought_process_object)
                print("Response -->>  ", response)
                print("🤖 Thought process -->>  ", thought_process_content, end="\n\n", flush=True)
                tool_calls = response.choices[0].message.tool_calls
                
                if not tool_calls:
                    print("✅ No more tools needed - analysis complete")
                    # emit('synthesis_progress', {
                    #     'session_id': session_id,
                    #     'message': "Tool analysis complete, no more tools needed",
                    #     'timestamp': datetime.now().isoformat()
                    # })
                    final_response = response.choices[0].message.content
                    final_response_object = {
                        "role": "assistant",
                        "content": final_response
                    }
                    session_history.append(final_response_object)
                    print("🤖 Final response -->>  ", final_response_object.get("content", "Final response"), end="\n\n", flush=True)
                    return final_response_object
                    # break
                
                print(f"🔧 {len(tool_calls)} tool(s) to execute: {[tc.function.name for tc in tool_calls]}")
                
                # Execute each tool call
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        tool_params = json.loads(tool_call.function.arguments)
                    except:
                        tool_params = {}
                    
                    print(f"🔧 Executing tool: {tool_name} with params: {tool_params}")
                    
                    # Emit tool start in real-time
                    # emit('tool_start', {
                    #     'session_id': session_id,
                    #     'tool_name': tool_name,
                    #     'description': f"Executing {tool_name}",
                    #     'parameters': tool_params,
                    #     'timestamp': datetime.now().isoformat(),
                    #     'status': 'running'
                    # })
                    
                    # Add tool action to history
                    tool_action_message = {
                        "role": "assistant",
                        "content": f"Executing {tool_name} with parameters: {tool_params}"
                    }
                    session_history.append(tool_action_message)
                    
                    # Execute the tool
                    try:
                        if tool_name == "csv_reader":
                            if not system:
                                system = {}
                            file_path = tool_params.get("file_path", "supply_chain_data.csv")
                            result = DataTools.csv_reader(file_path)
                            system['current_data'] = result
                            # execution_context.data_context['data_loaded'] = True
                            # execution_context.data_context['raw_data'] = result
                            # execution_context.data_context['products'] = result['product_name'].unique().tolist()
                            tool_result = {
                                "success": True,
                                "records_loaded": len(result),
                                "products": result['product_name'].unique().tolist()
                            }
                            
                        elif tool_name == "data_filter":
                            if 'current_data' not in system:
                                tool_result = {"error": "No data loaded. Please load data first."}
                            else:
                                result = DataTools.data_filter(system['current_data'], **tool_params)
                                system['current_data'] = result
                                tool_result = {
                                    "success": True,
                                    "filtered_records": len(result),
                                    "data_preview": result.head().to_dict() if len(result) > 0 else {}
                                }
                        
                        elif tool_name == "feature_engineer":
                            if 'current_data' not in system:
                                tool_result = {"error": "No data loaded. Please load data first."}
                            else:
                                result = DataTools.feature_engineer(system['current_data'], **tool_params)
                                system['current_data'] = result
                                # Extract stockout risk analysis
                                latest_data = result.iloc[-1] if len(result) > 0 else {}
                                stockout_risk = {
                                    "current_stock": latest_data.get('current_stock', 0),
                                    "stockout_risk": latest_data.get('stockout_risk', 0),
                                    "avg_daily_demand": latest_data.get('avg_daily_demand', 0)
                                }
                                # execution_context.analysis_results['stockout_risk_analysis'] = stockout_risk
                                tool_result = {
                                    "success": True,
                                    "stockout_risk_calculated": True,
                                    "latest_metrics": stockout_risk
                                }
                        
                        elif tool_name == "demand_forecaster":
                            if 'current_data' not in system:
                                tool_result = {"error": "No data loaded. Please load data first."}
                            else:
                                result = AnalyticsTools.demand_forecaster(system['current_data'], **tool_params)
                                # execution_context.analysis_results['demand_forecast'] = result
                                tool_result = {
                                    "success": True,
                                    "forecast_generated": True,
                                    "forecast_horizon": tool_params.get('horizon', 14),
                                    "forecast_data": result
                                }
                        
                        elif tool_name == "comprehensive_stockout_analysis":
                            if 'current_data' not in system:
                                tool_result = {"error": "No data loaded. Please load data first."}
                            else:
                                result = StockoutRiskAnalyzer.comprehensive_stockout_analysis(system['current_data'], **tool_params)
                                tool_result = {
                                    "success": True,
                                    "stockout_analysis_completed": True,
                                    "analysis_results": result,
                                    "products_analyzed": list(result.keys())
                                }
                        
                        elif tool_name == "analyze_supplier_performance":
                            if 'current_data' not in system:
                                tool_result = {"error": "No data loaded. Please load data first."}
                            else:
                                result = SupplierPerformanceAnalyzer.analyze_supplier_performance(system['current_data'])
                                tool_result = {
                                    "success": True,
                                    "supplier_analysis_completed": True,
                                    "performance_results": result,
                                    "suppliers_analyzed": list(result.keys())
                                }
                        
                        elif tool_name == "calculate_optimal_inventory_levels":
                            if 'current_data' not in system:
                                tool_result = {"error": "No data loaded. Please load data first."}
                            else:
                                result = InventoryOptimizationCalculator.calculate_optimal_inventory_levels(
                                    system['current_data'], 
                                    carrying_cost_rate=tool_params.get('carrying_cost_rate', 0.25),
                                    stockout_cost_multiplier=tool_params.get('stockout_cost_multiplier', 5)
                                )
                                tool_result = {
                                    "success": True,
                                    "optimization_completed": True,
                                    "optimization_results": result,
                                    "products_optimized": list(result.keys())
                                }
                        
                        elif tool_name == "analyze_financial_impact":
                            if 'current_data' not in system:
                                tool_result = {"error": "No data loaded. Please load data first."}
                            else:
                                result = FinancialImpactAnalyzer.analyze_financial_impact(
                                    system['current_data'],
                                    carrying_cost_rate=tool_params.get('carrying_cost_rate', 0.25),
                                    stockout_cost_multiplier=tool_params.get('stockout_cost_multiplier', 5),
                                    profit_margin=tool_params.get('profit_margin', 0.3)
                                )
                                tool_result = {
                                    "success": True,
                                    "financial_analysis_completed": True,
                                    "financial_results": result,
                                    "portfolio_summary": result.get("portfolio_summary", {})
                                }
                        
                        else:
                            tool_result = {"error": f"Unknown tool: {tool_name}"}
                        
                        # Store tool result
                        # all_tool_results.append({
                        #     "tool_name": tool_name,
                        #     "parameters": tool_params,
                        #     "result": tool_result,
                        #     "iteration": iteration_count
                        # })
                        
                        # # Emit tool completion in real-time
                        # emit('tool_complete', {
                        #     'session_id': session_id,
                        #     'tool_name': tool_name,
                        #     'summary': f"{tool_name} completed successfully",
                        #     'result_data': tool_result,
                        #     'timestamp': datetime.now().isoformat(),
                        #     'status': 'completed'
                        # })
                        
                        # Add tool result to history
                        tool_result_message = {
                            "role": "system",
                            "content": f"Tool {tool_name} completed successfully. Result: {json.dumps(tool_result, default=str)}"
                        }
                        session_history.append(tool_result_message)
                        
                    except Exception as e:
                        print(f"❌ Tool {tool_name} execution failed: {str(e)}")
                        import traceback
                        print(f"🔍 [TRACEBACK] Tool execution error: {traceback.format_exc()}")
                        error_result = {"error": f"Tool {tool_name} failed: {str(e)}"}
                        # all_tool_results.append({
                        #     "tool_name": tool_name,
                        #     "parameters": tool_params,
                        #     "result": error_result,
                        #     "iteration": iteration_count
                        # })
                        
                        # Add error to history
                        error_message = {
                            "role": "system", 
                            "content": f"Tool {tool_name} failed with error: {str(e)}"
                        }
                        session_history.append(error_message)
            




            print(f"Making a recursive call with session history object -->>  \n\n {session_history}")
            print("\n\n\n\n\n\n\n")
            final_resp = cls.decide_tools_and_generate_insights(user_query, session_history, session_id, system)
            
            return None
            
#             # Generate final response using ResponseSynthesizer
#             print("🎯 Generating final response using ResponseSynthesizer...")
#             print(f"📊 Tools executed: {len(all_tool_results)}")
            
#             # Emit synthesis start
#             emit('synthesis_start', {
#                 'session_id': session_id,
#                 'synthesis_type': 'response_synthesis',
#                 'timestamp': datetime.now().isoformat()
#             })
            
#             # Populate execution context with all results
#             execution_context.updated_at = datetime.now()
            
#             print(f"📊 Populating execution context with {len(all_tool_results)} tool results")
            
#             # Add data context from current session
#             if 'current_data' in system.context:
#                 df = system.context['current_data']
#                 execution_context.data_context.update({
#                     'total_records': len(df),
#                     'products': df['product_name'].unique().tolist() if 'product_name' in df.columns else [],
#                     'date_range': {
#                         'start': df['date'].min() if 'date' in df.columns else None,
#                         'end': df['date'].max() if 'date' in df.columns else None
#                     }
#                 })
#                 print(f"📈 Updated context with {len(df)} records")
            
#             # Emit progress update
#             emit('synthesis_progress', {
#                 'session_id': session_id,
#                 'message': "Preparing response synthesis...",
#                 'timestamp': datetime.now().isoformat()
#             })
            
#             try:
#                 print("🔮 Calling ResponseSynthesizer...")
#                 # Use ResponseSynthesizer to generate final response
#                 response_synthesizer = ResponseSynthesizer()
#                 final_response = response_synthesizer.synthesize_response(execution_context)
#                 print(f"✅ ResponseSynthesizer completed. Response length: {len(final_response) if final_response else 0}")
                
#                 if not final_response:
#                     print("⚠️ Empty response from synthesizer, generating fallback")
#                     final_response = f"""# Supply Chain Analysis Report

# ## Query Analysis
# Your request: {user_query}

# ## Tools Executed
# {chr(10).join([f"- {result['tool_name']}: {'✅ Success' if result['result'].get('success', True) else '❌ Failed'}" for result in all_tool_results])}

# ## Summary
# Analysis completed successfully with {len(all_tool_results)} tools executed.
# Processing time: {time.time() - start_time:.2f} seconds.

# *Note: Detailed analysis results are available in the execution context.*
# """
                
#             except Exception as e:
#                 print(f"❌ ResponseSynthesizer failed: {str(e)}")
#                 import traceback
#                 print(f"🔍 Synthesizer traceback: {traceback.format_exc()}")
                
#                 # Generate a fallback response
#                 final_response = f"""# Supply Chain Analysis Report

# ## Query Analysis
# Your request: {user_query}

# ## Tools Executed
# {chr(10).join([f"- {result['tool_name']}: {'✅ Success' if result['result'].get('success', True) else '❌ Failed'}" for result in all_tool_results])}

# ## Analysis Results
# The analysis was completed but response synthesis encountered an error: {str(e)}

# ## Summary
# - Processing completed in {time.time() - start_time:.2f} seconds
# - {len(all_tool_results)} tools executed successfully
# - Data analysis available in execution context

# *Please try rephrasing your query for better results.*
# """
                
#                 emit('error', {
#                     'session_id': session_id,
#                     'error_type': 'synthesis_error',
#                     'message': f"Response synthesis failed: {str(e)}",
#                     'timestamp': datetime.now().isoformat()
#                 })
            
#             print(f"📝 Final response ready ({len(final_response)} characters)")
            
#             # Add final response to session history
#             final_response_message = {
#                 "role": "assistant",
#                 "content": final_response
#             }
#             session_history.append(final_response_message)
            
#             # Stream the response in real-time
#             try:
#                 print("📡 Starting response streaming...")
#                 emit('message', {
#                     'streaming_status': 'message_start',
#                     'session_id': session_id,
#                     'timestamp': datetime.now().isoformat()
#                 })
                
#                 # Stream word by word for real-time effect
#                 words = final_response.split(' ')
#                 current_content = ''
#                 for i, word in enumerate(words):
#                     current_content += word + ' '
#                     emit('message', {
#                         'streaming_status': 'message_continue',
#                         'text_response': current_content.strip(),
#                         'session_id': session_id,
#                         'timestamp': datetime.now().isoformat()
#                     })
#                     # Small delay for streaming effect
#                     time.sleep(0.02)
                
#                 # End streaming
#                 emit('message', {
#                     'streaming_status': 'end',
#                     'text_response': final_response,
#                     'session_id': session_id,
#                     'timestamp': datetime.now().isoformat()
#                 })
#                 print("✅ Response streaming completed")
                
#             except Exception as e:
#                 print(f"❌ Failed to emit response stream: {e}")
            
#             # Prepare tools_used array for frontend
#             tools_used = []
#             for tool_result in all_tool_results:
#                 tools_used.append({
#                     'name': tool_result['tool_name'],
#                     'execution_time': 0.5,  # Default execution time
#                     'success': tool_result['result'].get('success', True)
#                 })
            
#             # Calculate total processing time
#             processing_time = time.time() - start_time
            
#             # Emit execution metrics in real-time
#             emit('execution_metrics', {
#                 'session_id': session_id,
#                 'processing_time': processing_time,
#                 'tools_used': tools_used,
#                 'success_rate': 100,
#                 'timestamp': datetime.now().isoformat()
#             })
            
#             # Emit processing complete
#             try:
#                 emit('processing_complete', {
#                     'session_id': session_id,
#                     'success': True,
#                     'processing_time': processing_time,
#                     'timestamp': datetime.now().isoformat()
#                 })
#             except Exception as e:
#                 print(f"❌ Failed to emit processing_complete: {e}")
            
#             print("✅ Analysis complete, response generated and saved to history")
            
#             return {
#                 'response': final_response,
#                 'processing_time': processing_time,
#                 'tools_used': tools_used,
#                 'success': True
#             }
            
        except TimeoutError as e:
            # Handle timeout
            processing_time = time.time() - start_time if 'start_time' in locals() else 0
            timeout_response = f"⏰ Analysis timed out after {processing_time:.1f} seconds. The system was taking too long to process your query. Please try with a simpler request or contact support."
            
            print(f"⏰ [TIMEOUT] Analysis timed out after {processing_time:.1f}s")
            import traceback
            print(f"🔍 [TRACEBACK] Timeout error: {traceback.format_exc()}")
            
            emit('error', {
                'session_id': session_id,
                'error_type': 'timeout_error',
                'message': "Analysis timed out",
                'timestamp': datetime.now().isoformat()
            })
            
            return {
                'response': timeout_response,
                'processing_time': processing_time,
                'tools_used': [],
                'success': False,
                'error': 'timeout'
            }
            
        except Exception as e:
            print(f"❌ [ERROR] Exception in decide_tools_and_generate_insights: {str(e)}")
            import traceback
            print(f"🔍 [TRACEBACK] {traceback.format_exc()}")
            
            # Fallback response
            processing_time = time.time() - start_time if 'start_time' in locals() else 0
            fallback_response = f"I apologize, but I encountered an error while processing your query: {user_query}. Please try rephrasing your question or contact support."
            
            # Add fallback to history if session exists
            try:
                if 'system' in locals() and system:
                    system.session.add_message_to_history({
                        "role": "assistant",
                        "content": fallback_response
                    })
            except Exception as nested_e:
                print(f"❌ Failed to add fallback to history: {nested_e}")
                import traceback
                print(f"🔍 [TRACEBACK] Fallback history error: {traceback.format_exc()}")
            
            return {
                'response': fallback_response,
                'processing_time': processing_time,
                'tools_used': [],
                'success': False,
                'error': str(e)
            }
            
        finally:
            # Cleanup handled automatically with thread-safe approach
            pass




# ----------------------------
# Module 6: Execution Engine - Flask-Ready Session Management
# ----------------------------
class AgenticSystem:
    def __init__(self, data_path="supply_chain_data.csv", session_id=None):
        """Initialize with mandatory client-provided session ID"""
        if not session_id:
            raise ValueError("Session ID must be provided by client (Flask)")
        
        self.data_path = data_path
        self.context = {}
        self.session = SessionManager(session_id)
    
    def execute_tool(self, tool_name, params):
        """Execute tool based on name and parameters"""
        if tool_name == "data_filter":
            return DataTools.data_filter(self.context['current_data'], **params)
        elif tool_name == "feature_engineer":
            return DataTools.feature_engineer(self.context['current_data'], **params)
        elif tool_name == "demand_forecaster":
            return AnalyticsTools.demand_forecaster(self.context['current_data'], **params)
        return None

    def run_query(self, user_query):
        """Main execution flow with enhanced message history management"""
        self.session.log_query_start(user_query)
        print(f"🤖 Processing: {user_query}\n")
        
        # Create progress bar for overall process
        steps = ["Loading Data", "Tool Selection", "Data Filtering", "Risk Analysis", "Forecasting", "Generating Insights"]
        
        with tqdm(total=len(steps), desc="Analysis Progress", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        
            # Step 1: Load data
            pbar.set_description("Loading inventory data")
            if 'current_data' not in self.context:
                time.sleep(1)
                self.context['current_data'] = DataTools.csv_reader(self.data_path)
            pbar.update(1)
        
            # Step 2: Tool selection
            pbar.set_description("Selecting analysis tools")
            time.sleep(0.8)
            tool_call = Orchestrator.decide_tools(user_query)
            pbar.update(1)
        
            # Step 3: Data filtering and analysis
            pbar.set_description("Filtering product data")
            time.sleep(0.6)
            if "Bluetooth Speaker" in user_query:
                filtered_data = DataTools.data_filter(
                    self.context['current_data'], 
                    product_name="Bluetooth Speaker"
                )
                self.context['current_data'] = filtered_data
            
                # Add feature engineering
                print(" -> Calculating risk metrics...")
                self.context['current_data'] = DataTools.feature_engineer(self.context['current_data'])
            
                # Generate forecast
                print(" -> Generating demand forecast...")
                forecast = AnalyticsTools.demand_forecaster(self.context['current_data'], horizon=14)
            
                # Business analysis
                print(" -> Analyzing stockout risk and reorder recommendations...")
                product_data = self.context['current_data'].iloc[-1]
                stockout = AnalyticsTools.stockout_risk_assessor(
                    forecast,
                    product_data['current_stock'],
                    product_data['lead_time_days']
                )
                reorder = BusinessTools.replenishment_advisor(
                    product_data['current_stock'],
                    forecast,
                    product_data['unit_cost']
                )
                self.context['analysis'] = {
                    "product": "Bluetooth Speaker",
                    "forecast": forecast,
                    "stockout": stockout, 
                    "reorder": reorder,
                    "current_stock": product_data['current_stock'],
                    "avg_daily_demand": product_data.get('avg_daily_demand', 'N/A')
                }
            pbar.update(1)
        
            # Step 4: Generate insights using new message structure
            pbar.set_description("Generating insights")
            time.sleep(1)
            print(" -> Generating executive insights...")
            pbar.update(1)
        
        print("\n✅ Analysis complete!")
        
        # Generate insights with enhanced message history and analysis context
        result = Orchestrator.generate_insights(
            self.session.session_id,
            user_query,
            self.context.get('analysis', {})
        )
        
        # Log the completed query
        self.session.log_query_end(result)
        return result

# ----------------------------
# Redis-Based Session Management Factory
# ----------------------------
class SessionFactory:
    """Factory class to manage multiple client sessions using Redis"""
    
    @classmethod
    def get_or_create_session(cls, session_id, data_path="supply_chain_data.csv"):
        """Get existing session or create new one with client-provided session_id"""
        if not session_id:
            raise ValueError("Session ID must be provided by client")
        
        # Check if session exists in Redis
        session_key = f"session:{session_id}"
        if redis_client.exists(session_key):
            print(f"📋 Using existing session: {session_id}")
        else:
            print(f"📋 Created new session: {session_id}")
        
        return AgenticSystem(data_path=data_path, session_id=session_id)
    
    @classmethod
    def get_session(cls, session_id):
        """Get existing session by ID"""
        if not session_id:
            return None
        
        session_key = f"session:{session_id}"
        if redis_client.exists(session_key):
            return AgenticSystem(data_path="supply_chain_data.csv", session_id=session_id)
        return None
    
    @classmethod
    def session_exists(cls, session_id):
        """Check if session exists"""
        if not session_id:
            return False
        session_key = f"session:{session_id}"
        return redis_client.exists(session_key)
    
    @classmethod
    def end_session(cls, session_id):
        """End and cleanup session"""
        if not session_id:
            return None
        
        session_key = f"session:{session_id}"
        if redis_client.exists(session_key):
            # Get session data for summary
            session_data = pickle.loads(redis_client.get(session_key))
            
            # Create summary
            session_end = datetime.now()
            session_start = datetime.fromisoformat(session_data["session_start"])
            session_duration = session_end - session_start
            
            summary = {
                "session_id": session_data["session_id"],
                "started": session_start.strftime('%Y-%m-%d %H:%M:%S'),
                "ended": session_end.strftime('%Y-%m-%d %H:%M:%S'),
                "total_duration": str(session_duration).split('.')[0],
                "queries_processed": session_data["query_count"],
                "total_processing_time": f"{session_data['total_processing_time']:.2f}s",
                "avg_query_time": f"{session_data['total_processing_time']/max(1, session_data['query_count']):.2f}s",
                "total_messages": len(session_data["message_history"])
            }
            
            # Delete session from Redis
            redis_client.delete(session_key)
            print(f"📋 Ended session: {session_id}")
            return summary
        return None
    
    @classmethod
    def cleanup_expired_sessions(cls, timeout_minutes=30):
        """Clean up expired sessions based on inactivity"""
        # Redis TTL handles expiration automatically, but we can manually check
        session_keys = redis_client.keys("session:*")
        expired_sessions = []
        summaries = []
        
        for key in session_keys:
            try:
                session_data = pickle.loads(redis_client.get(key))
                last_activity = datetime.fromisoformat(session_data["last_activity"])
                idle_time = datetime.now() - last_activity
                
                if idle_time.total_seconds() > (timeout_minutes * 60):
                    session_id = session_data["session_id"]
                    expired_sessions.append(session_id)
                    summary = cls.end_session(session_id)
                    if summary:
                        summaries.append(summary)
            except:
                # If we can't parse the session data, delete the key
                redis_client.delete(key)
        
        if expired_sessions:
            print(f"🧹 Cleaned up {len(expired_sessions)} expired sessions")
        
        return summaries
    
    @classmethod
    def get_all_sessions(cls):
        """Get all active sessions info"""
        session_keys = redis_client.keys("session:*")
        sessions_info = {}
        
        for key in session_keys:
            try:
                session_data = pickle.loads(redis_client.get(key))
                session_id = session_data["session_id"]
                
                # Create session manager to get info
                temp_session = SessionManager(session_id)
                sessions_info[session_id] = temp_session.get_session_info()
            except:
                # If we can't parse the session data, skip it
                continue
        
        return sessions_info
    
    @classmethod
    def get_session_count(cls):
        """Get number of active sessions"""
        session_keys = redis_client.keys("session:*")
        return len(session_keys)
    
    @classmethod
    def get_session_stats(cls):
        """Get overall session statistics"""
        session_keys = redis_client.keys("session:*")
        
        if not session_keys:
            return {
                "total_sessions": 0,
                "active_sessions": 0,
                "total_queries": 0,
                "avg_queries_per_session": 0
            }
        
        total_queries = 0
        valid_sessions = 0
        
        for key in session_keys:
            try:
                session_data = pickle.loads(redis_client.get(key))
                total_queries += session_data["query_count"]
                valid_sessions += 1
            except:
                continue
        
        return {
            "total_sessions": valid_sessions,
            "active_sessions": valid_sessions,
            "total_queries": total_queries,
            "avg_queries_per_session": round(total_queries / max(1, valid_sessions), 2)
        }

# ----------------------------
# Flask Integration Helper Functions
# ----------------------------
def process_flask_query(session_id, user_query, data_path="supply_chain_data.csv"):
    """
    Main function for Flask integration
    Handles session management and query processing
    """
    try:
        # Get or create session
        system = SessionFactory.get_or_create_session(session_id, data_path)
        
        # Process query
        result = system.run_query(user_query)
        
        return {
            "success": True,
            "session_id": session_id,
            "response": result,
            "session_info": system.session.get_session_info()
        }
    
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"❌ [ERROR] Exception in process_flask_query: {str(e)}")
        print(f"🔍 [TRACEBACK] {error_traceback}")
        
        return {
            "success": False,
            "session_id": session_id,
            "error": str(e),
            "error_type": type(e).__name__,
            "response": f"Error processing query: {str(e)}",
            "traceback": error_traceback
        }
# class SessionManager:
#     # ... existing methods ...
    
#     def debug_session_data(self):
#         """Debug method to inspect session data in Redis"""
#         data = self._get_session_data()
#         if data:
#             print(f"Session ID: {data['session_id']}")
#             print(f"Query Count: {data['query_count']}")
#             print(f"Message History Length: {len(data['message_history'])}")
#             print("Message History:")
#             for i, msg in enumerate(data['message_history']):
#                 role = msg.get('role', 'unknown')
#                 content_preview = str(msg.get('content', ''))[:100] + "..." if len(str(msg.get('content', ''))) > 100 else str(msg.get('content', ''))
#                 print(f"  {i}: {role} - {content_preview}")
#         else:
#             print("No session data found in Redis")

query  = input("Ask a question here: ")

print(Orchestrator.decide_tools_and_generate_insights(query, [], "123", {}))