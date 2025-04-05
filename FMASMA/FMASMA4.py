from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List
import json
import math

class Trader:
    def run(self, state: TradingState):
        # Deserialize the stored state from traderData (if available)
        try:
            stored_data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            stored_data = {}

        # Ensure that each product has its own list for storing mid-prices
        for product in ["RAINFOREST_RESIN", "KELP"]:
            if product not in stored_data:
                stored_data[product] = []

        # Define parameters for the moving averages and trading
        FAST_WINDOW = 3
        SLOW_WINDOW = 5
        TRADE_SIZE = 25
        POSITION_LIMIT = 50

        def compute_mid_price(order_depth: OrderDepth) -> float:
            # Compute the mid-price from the best bid and ask.
            if not order_depth.buy_orders or not order_depth.sell_orders:
                return None
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2.0

        result = {}

        for product in ["RAINFOREST_RESIN", "KELP"]:
            # If no order depth is available for the product, do nothing
            if product not in state.order_depths:
                result[product] = []
                continue

            order_depth: OrderDepth = state.order_depths[product]
            current_position = state.position.get(product, 0)
            mid_price = compute_mid_price(order_depth)
            if mid_price is None:
                result[product] = []
                continue

            # Update stored mid-prices for this product and keep only the latest SLOW_WINDOW values
            stored_data[product].append(mid_price)
            stored_data[product] = stored_data[product][-SLOW_WINDOW:]

            # Only compute moving averages when we have enough data points
            if len(stored_data[product]) < SLOW_WINDOW:
                result[product] = []
                continue

            # Calculate the fast (short-term) and slow (long-term) moving averages
            fast_ma = sum(stored_data[product][-FAST_WINDOW:]) / FAST_WINDOW
            slow_ma = sum(stored_data[product]) / SLOW_WINDOW

            orders: List[Order] = []

            # Bullish signal: fast MA is above slow MA --> Buy at the best ask price
            if fast_ma > slow_ma:
                best_ask = min(order_depth.sell_orders.keys())
                if current_position + TRADE_SIZE <= POSITION_LIMIT:
                    orders.append(Order(product, best_ask, TRADE_SIZE))
            # Bearish signal: fast MA is below slow MA --> Sell at the best bid price
            elif fast_ma < slow_ma:
                best_bid = max(order_depth.buy_orders.keys())
                if current_position - TRADE_SIZE >= -POSITION_LIMIT:
                    orders.append(Order(product, best_bid, -TRADE_SIZE))
            
            result[product] = orders

        new_trader_data = json.dumps(stored_data)
        conversions = 0  # No conversion logic is applied in this simple strategy
        return result, conversions, new_trader_data
