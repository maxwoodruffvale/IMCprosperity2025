from datamodel import Order, OrderDepth, TradingState
from typing import List, Dict
import json
import numpy as np

class Trader:
    def __init__(self):
        # Define strategy parameters
        self.LOOKBACK_PERIOD = 20  # Number of periods for Donchian Channel
        self.ATR_PERIOD = 14       # Number of periods for ATR calculation
        self.TRADE_SIZE = 25       # Number of units to trade
        self.POSITION_LIMIT = 50   # Maximum position size

    def run(self, state: TradingState):
        # Load historical data from traderData
        try:
            stored_data = json.loads(state.traderData) if state.traderData else {}
        except json.JSONDecodeError:
            stored_data = {}

        # Initialize historical price storage if not present
        for product in ["RAINFOREST_RESIN", "KELP"]:
            if product not in stored_data:
                stored_data[product] = []

        # Helper function to compute mid-price
        def compute_mid_price(order_depth: OrderDepth) -> float:
            if not order_depth.buy_orders or not order_depth.sell_orders:
                return None
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2.0

        # Prepare result containers
        result = {}
        conversions = 0

        # Process each product
        for product in ["RAINFOREST_RESIN", "KELP"]:
            if product not in state.order_depths:
                continue

            order_depth = state.order_depths[product]
            current_position = state.position.get(product, 0)

            # Compute current mid-price
            mid_price = compute_mid_price(order_depth)
            if mid_price is None:
                result[product] = []
                continue

            # Update historical prices - remove oldest data if exceeding lookback period
            stored_data[product].append(mid_price)
            if len(stored_data[product]) > self.LOOKBACK_PERIOD:
                stored_data[product].pop(0)

            # Ensure sufficient data for calculations
            if len(stored_data[product]) < self.LOOKBACK_PERIOD:
                result[product] = []
                continue

            # Calculate Donchian Channel
            highest_high = max(stored_data[product])
            lowest_low = min(stored_data[product])

            # Calculate ATR for stop-loss (simplified)
            price_diffs = np.diff(stored_data[product])
            atr = np.mean(np.abs(price_diffs[-self.ATR_PERIOD:]))

            # Determine entry and exit signals
            orders: List[Order] = []

            # Entry signals
            if mid_price > highest_high:
                # Potential long entry
                if current_position < self.POSITION_LIMIT:
                    trade_size = min(self.TRADE_SIZE, self.POSITION_LIMIT - current_position)
                    # place an order to buy at the mid price
                    orders.append(Order(product, int(mid_price), trade_size))
            elif mid_price < lowest_low:
                # Potential short entry
                if current_position > -self.POSITION_LIMIT:
                    trade_size = min(self.TRADE_SIZE, self.POSITION_LIMIT + current_position)
                    # place an order to sell at the mid price
                    orders.append(Order(product, int(mid_price), -trade_size))

            # Exit signals (simplified: exit when price crosses opposite band)
            if current_position > 0 and mid_price < lowest_low:
                # Exit long positions
                orders.append(Order(product, int(mid_price), -current_position))
            elif current_position < 0 and mid_price > highest_high:
                # Exit short positions
                orders.append(Order(product, int(mid_price), -current_position))

            # Implement stop-loss (optional, based on ATR)
            stop_loss_price = None
            if current_position > 0:
                stop_loss_price = mid_price - 2 * atr
                if mid_price <= stop_loss_price:
                    orders.append(Order(product, int(mid_price), -current_position))
            elif current_position < 0:
                stop_loss_price = mid_price + 2 * atr
                if mid_price >= stop_loss_price:
                    orders.append(Order(product, int(mid_price), -current_position))

            result[product] = orders

        # Serialize updated historical data
        new_trader_data = json.dumps(stored_data)

        # Return orders, conversions, and updated trader data
        return result, conversions, new_trader_data
