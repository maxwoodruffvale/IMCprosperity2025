from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import jsonpickle
import numpy as np

class Trader:
    def __init__(self):
        self.window_size = 5  # Number of past prices to consider for SMA
        self.price_history = {}  # Dictionary to store price history for each product

    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        # Deserialize traderData to maintain state across runs
        if state.traderData:
            self.price_history = jsonpickle.decode(state.traderData)
        else:
            self.price_history = {}

        result = {}
        for product, order_depth in state.order_depths.items():
            orders: List[Order] = []

            # Update price history with the mid-price of the current order book
            if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                mid_price = (best_bid + best_ask) / 2
                if product not in self.price_history:
                    self.price_history[product] = []
                self.price_history[product].append(mid_price)

                # Maintain only the latest 'window_size' prices
                if len(self.price_history[product]) > self.window_size:
                    self.price_history[product].pop(0)

            # Calculate SMA if we have enough data points
            if len(self.price_history[product]) == self.window_size:
                sma = np.mean(self.price_history[product])
                print(f"SMA for {product}: {sma}")

                # Trading logic based on SMA
                # If the current mid price is below the SMA, it may indicate an upward trend
                if mid_price < sma:
                    # Place a buy order at the best ask price
                    best_ask_price = min(order_depth.sell_orders.keys())
                    best_ask_volume = order_depth.sell_orders[best_ask_price]
                    orders.append(Order(product, best_ask_price, best_ask_volume))
                    print(f"Placing BUY order for {product}: {best_ask_volume} units at {best_ask_price}")

                # If the current mid price is above the SMA, it may indicate a downward trend
                elif mid_price > sma: # flipped
                    # Place a sell order at the best bid price
                    best_bid_price = max(order_depth.buy_orders.keys())
                    best_bid_volume = order_depth.buy_orders[best_bid_price]
                    orders.append(Order(product, best_bid_price, -best_bid_volume)) # maybe remove - here
                    print(f"Placing SELL order for {product}: {best_bid_volume} units at {best_bid_price}")

            result[product] = orders

        # Serialize price history to maintain state in the next run
        traderData = jsonpickle.encode(self.price_history)

        conversions = 0  # No conversion requests in this strategy
        return result, conversions, traderData
