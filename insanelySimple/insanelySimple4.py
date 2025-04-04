from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import jsonpickle
import statistics

class Trader:
    
    def run(self, state: TradingState):
        # Load persistent state from traderData; if unavailable, initialize an empty dictionary.
        try:
            stored_data = jsonpickle.decode(state.traderData)
            if not isinstance(stored_data, dict):
                stored_data = {}
        except Exception:
            stored_data = {}
        
        result = {}
        # Define a small epsilon to detect stability in price (thus forcing trades)
        epsilon = 0.1
        
        for product, order_depth in state.order_depths.items():
            orders: List[Order] = []
            
            # Determine the best bid and best ask prices and volumes.
            best_bid = None
            best_bid_qty = None
            best_ask = None
            best_ask_qty = None
            
            if order_depth.buy_orders:
                sorted_bids = sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)
                best_bid, best_bid_qty = sorted_bids[0]
            
            if order_depth.sell_orders:
                sorted_asks = sorted(order_depth.sell_orders.items(), key=lambda x: x[0])
                best_ask, best_ask_qty = sorted_asks[0]
            
            # Compute the current mid price based on available orders.
            if best_bid is not None and best_ask is not None:
                mid_price = (best_bid + best_ask) / 2
            elif best_bid is not None:
                mid_price = best_bid
            elif best_ask is not None:
                mid_price = best_ask
            else:
                # If no order info is available for the product, skip it.
                continue
            
            # Retrieve or initialize historical mid prices for the product.
            if product not in stored_data:
                stored_data[product] = []
            historical_prices = stored_data[product]
            historical_prices.append(mid_price)
            # Retain only the last 5 mid prices.
            if len(historical_prices) > 5:
                historical_prices = historical_prices[-5:]
            stored_data[product] = historical_prices
            
            moving_avg = statistics.mean(historical_prices)
            
            # Decision logic:
            # - If current mid price is below the moving average, assume undervalued => BUY.
            # - If current mid price is above the moving average, assume overvalued => SELL.
            # - If the price is nearly equal to the moving average (stable), place both BUY and SELL orders to force activity.
            if abs(mid_price - moving_avg) < epsilon:
                # Stable market: place both buy and sell orders.
                if best_ask is not None and best_ask_qty is not None:
                    orders.append(Order(product, best_ask, -abs(best_ask_qty)))
                if best_bid is not None and best_bid_qty is not None:
                    orders.append(Order(product, best_bid, abs(best_bid_qty)))
            elif mid_price < moving_avg:
                # Market is below average; attempt to buy.
                if best_ask is not None and best_ask_qty is not None:
                    orders.append(Order(product, best_ask, -abs(best_ask_qty)))
            elif mid_price > moving_avg:
                # Market is above average; attempt to sell.
                if best_bid is not None and best_bid_qty is not None:
                    orders.append(Order(product, best_bid, abs(best_bid_qty)))
            
            result[product] = orders
        
        # Save persistent state for future iterations.
        traderData = jsonpickle.encode(stored_data)
        conversions = 0
        
        return result, conversions, traderData
