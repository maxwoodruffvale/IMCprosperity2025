from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import jsonpickle

class Trader:
    
    def run(self, state: TradingState):
        # Retrieve persistent state from traderData if available, else initialize as empty dict.
        try:
            stored_data = jsonpickle.decode(state.traderData)
            if not isinstance(stored_data, dict):
                stored_data = {}
        except Exception:
            stored_data = {}
        
        result = {}
        
        # Loop over each product in the current order depth.
        for product, order_depth in state.order_depths.items():
            orders: List[Order] = []
            
            # Sort buy orders in descending order to get the best bid.
            best_bid = None
            best_bid_qty = None
            if order_depth.buy_orders:
                sorted_bids = sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)
                best_bid, best_bid_qty = sorted_bids[0]
            
            # Sort sell orders in ascending order to get the best ask.
            best_ask = None
            best_ask_qty = None
            if order_depth.sell_orders:
                sorted_asks = sorted(order_depth.sell_orders.items(), key=lambda x: x[0])
                best_ask, best_ask_qty = sorted_asks[0]
            
            acceptable_price = None
            margin = 1  # A fixed margin unit; can be adjusted based on product volatility
            
            if best_bid is not None and best_ask is not None:
                # Compute mid price from best bid and best ask.
                mid_price = (best_bid + best_ask) / 2
                acceptable_price = mid_price
                
                # If the best ask is significantly lower than the mid price, it indicates a potential undervaluation.
                if best_ask < mid_price - margin:
                    # Place a buy order: use the ask price and buy the full available quantity.
                    orders.append(Order(product, best_ask, -abs(best_ask_qty)))
                
                # If the best bid is significantly higher than the mid price, it indicates a potential overvaluation.
                elif best_bid > mid_price + margin:
                    # Place a sell order: use the bid price and sell the full available quantity.
                    orders.append(Order(product, best_bid, abs(best_bid_qty)))
            
            # If only one side of the market is available, use the stored price for that product (or fallback to current best price)
            elif best_bid is not None:
                acceptable_price = best_bid
                prev_price = stored_data.get(product, best_bid)
                # If the current bid is higher than the previous acceptable price, try selling.
                if best_bid > prev_price:
                    orders.append(Order(product, best_bid, abs(best_bid_qty)))
            
            elif best_ask is not None:
                acceptable_price = best_ask
                prev_price = stored_data.get(product, best_ask)
                # If the current ask is lower than the previous acceptable price, try buying.
                if best_ask < prev_price:
                    orders.append(Order(product, best_ask, -abs(best_ask_qty)))
            
            # Save orders for the current product.
            result[product] = orders
            
            # Update our stored state with the current acceptable price.
            if acceptable_price is not None:
                stored_data[product] = acceptable_price
        
        # Serialize stored state using jsonpickle so it persists for future iterations.
        traderData = jsonpickle.encode(stored_data)
        # No conversion logic is applied in this simple strategy.
        conversions = 0
        
        return result, conversions, traderData
