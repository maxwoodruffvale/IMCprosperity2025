from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List
import json
import math

class Trader:
    def run(self, state: TradingState):
        try:
            stored_data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            stored_data = {}

        # Ensure each product has a stored price history list.
        for product in ["RAINFOREST_RESIN", "KELP"]:
            if product not in stored_data:
                stored_data[product] = []

        # Define moving average windows and trade parameters.
        FAST_WINDOW = 3
        SLOW_WINDOW = 7
        TRADE_SIZE = 25
        POSITION_LIMIT = 50

        # Function to compute mid price from order depth.
        def compute_mid_price(order_depth: OrderDepth) -> float:
            if not order_depth.buy_orders or not order_depth.sell_orders:
                return None
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2.0

        result = {}

        for product in ["RAINFOREST_RESIN", "KELP"]:
            if product not in state.order_depths:
                continue

            order_depth: OrderDepth = state.order_depths[product]
            current_position = state.position.get(product, 0)
            mid_price = compute_mid_price(order_depth)
            if mid_price is None:
                result[product] = []
                continue
            
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())

            # Update stored price history and trim to at most SLOW_WINDOW prices.
            stored_data[product].append(mid_price)
            if len(stored_data[product]) > SLOW_WINDOW:
                stored_data[product] = stored_data[product][-SLOW_WINDOW:]

            orders: List[Order] = []
            # Only generate signals if we have enough data to compute both averages.
            if len(stored_data[product]) >= SLOW_WINDOW: # maybe make less strict
                # Compute moving averages.
                fast_avg = sum(stored_data[product][-FAST_WINDOW:]) / FAST_WINDOW
                slow_avg = sum(stored_data[product][-SLOW_WINDOW:]) / SLOW_WINDOW

                # Bullish signal: fast average is above slow average -> Buy.
                if fast_avg > slow_avg:
                    if current_position < POSITION_LIMIT and order_depth.sell_orders:
                        best_ask = min(order_depth.sell_orders.keys())
                        # Sell orders have negative quantities; take the absolute value.
                        available_qty = abs(order_depth.sell_orders.get(best_ask, 0))
                        qty_to_buy = min(TRADE_SIZE, POSITION_LIMIT - current_position, available_qty)
                        if qty_to_buy > 0:
                            orders.append(Order(product, best_ask, qty_to_buy))

                # Bearish signal: fast average is below slow average -> Sell.
                elif fast_avg < slow_avg:
                    if current_position > -POSITION_LIMIT and order_depth.buy_orders:
                        best_bid = max(order_depth.buy_orders.keys())
                        available_qty = order_depth.buy_orders.get(best_bid, 0)
                        qty_to_sell = min(TRADE_SIZE, current_position + POSITION_LIMIT, available_qty)
                        if qty_to_sell > 0:
                            orders.append(Order(product, best_bid, -qty_to_sell))

            result[product] = orders

        new_trader_data = json.dumps(stored_data)
        conversions = 0
        #logger.flush(state, result, conversions, new_trader_data)
        return result, conversions, new_trader_data
