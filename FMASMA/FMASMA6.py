from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List
import json
import math

class Trader:
    def run(self, state: TradingState):
        try:
            stored_data = json.loads(state.traderData) if state.traderData else {}
        except:
            stored_data = {}

        if "RAINFOREST_RESIN" not in stored_data:
            stored_data["RAINFOREST_RESIN"] = []
        if "KELP" not in stored_data:
            stored_data["KELP"] = []

        SMA_WINDOW = 7 # previsoulsy 5 for standalone results
        FAST_WINDOW = 3
        TRADE_SIZE = 25
        POSITION_LIMIT = 50

        def compute_mid_price(order_depth: OrderDepth) -> float:
            if len(order_depth.buy_orders) == 0 or len(order_depth.sell_orders) == 0:
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

            stored_data[product].append(mid_price)
            stored_data[product] = stored_data[product][-SMA_WINDOW:]

            sma = sum(stored_data[product]) / len(stored_data[product])

            fast_ma = sum(stored_data[product][-FAST_WINDOW:]) / FAST_WINDOW

            orders: List[Order] = []

            limit = POSITION_LIMIT
            size = TRADE_SIZE


            if fast_ma > sma:
                while current_position + size <= limit and best_ask < sma: # issue a buy order
                    orders.append(Order(product, best_ask, size))
                    limit -= size
                    best_ask += 1
                    try:
                        size = order_depth.sell_orders[best_ask]
                    except:
                        size = 0
                orders.append(Order(product, best_ask, POSITION_LIMIT - current_position))

            """
            while best_ask < sma and fast_ma > sma: # issue a buy order
                if current_position + size <= limit:
                    orders.append(Order(product, best_ask, size))
                    limit -= size
                    best_ask += 1
                    try:
                        size = order_depth.sell_orders[best_ask]
                    except:
                        size = 0
                else:
                    orders.append(Order(product, best_ask, POSITION_LIMIT - current_position))
                    break
            """

            limit = POSITION_LIMIT
            size = TRADE_SIZE

            if fast_ma < sma:
                while current_position - size >= -limit and best_bid > sma: # issue a sell order
                    orders.append(Order(product, best_bid, -size))
                    limit += size
                    best_bid -= 1
                    try:
                        size = order_depth.sell_orders[best_bid]
                    except:
                        size = 0
                orders.append(Order(product, best_bid, -POSITION_LIMIT + current_position))

            """
            while best_bid > sma and sma > fast_ma: # issue a sell order
                if current_position - size >= -limit:
                    orders.append(Order(product, best_bid, -size))
                    limit += size
                    best_bid -= 1
                    try:
                        size = order_depth.buy_orders[best_bid]
                    except:
                        size = 0
                else:
                    orders.append(Order(product, best_bid, -POSITION_LIMIT + current_position))
                    break
            """
            result[product] = orders

        new_trader_data = json.dumps(stored_data)
        conversions = 0
        return result, conversions, new_trader_data