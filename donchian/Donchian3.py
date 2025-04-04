import jsonpickle
from datamodel import OrderDepth, TradingState, Order, Trade
from typing import List, Dict

LOOKBACK_PERIOD = 20  # Number of iterations to compute the channel

class Trader:
    
    def run(self, state: TradingState):
        # Load persistent history from traderData (or initialize if not available)
        try:
            history = jsonpickle.decode(state.traderData)
            if not isinstance(history, dict):
                history = {}
        except Exception:
            history = {}
        
        result: Dict[str, List[Order]] = {}
        conversions = 0  # No conversion logic in this implementation
        
        # Process each product from the order depths
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            
            # Determine current price: prefer the last market trade price if available
            current_price = None
            if product in state.market_trades and state.market_trades[product]:
                current_price = state.market_trades[product][-1].price
            else:
                # Otherwise, derive the mid-price from the best bid and ask
                best_bid = self.get_best_bid(order_depth)
                best_ask = self.get_best_ask(order_depth)
                if best_bid is not None and best_ask is not None:
                    current_price = (best_bid + best_ask) / 2
            
            if current_price is None:
                result[product] = orders
                continue
            
            # Update the price history for the product
            if product not in history:
                history[product] = []
            history[product].append(current_price)
            if len(history[product]) > LOOKBACK_PERIOD:
                history[product].pop(0)
            
            print("Product:", product)
            print("Current Price:", current_price)
            print("Price History (len={}):".format(len(history[product])), history[product])
            
            # Only attempt trading if we have enough history to compute the channel
            if len(history[product]) == LOOKBACK_PERIOD:
                channel_high = max(history[product])
                channel_low = min(history[product])
                print("Channel High:", channel_high, "Channel Low:", channel_low)
                
                # Breakout upward: current price >= channel high, so we BUY using the best ask price.
                if current_price >= channel_high:
                    best_ask = self.get_best_ask(order_depth)
                    if best_ask is not None:
                        best_ask_amount = order_depth.sell_orders[best_ask]
                        print("Breakout UP! Placing BUY order for", -best_ask_amount, "units at", best_ask)
                        # Using negative volume for sell orders in the depth makes this a BUY order.
                        orders.append(Order(product, best_ask, -best_ask_amount))
                
                # Breakout downward: current price <= channel low, so we SELL using the best bid price.
                elif current_price <= channel_low:
                    best_bid = self.get_best_bid(order_depth)
                    if best_bid is not None:
                        best_bid_amount = order_depth.buy_orders[best_bid]
                        print("Breakout DOWN! Placing SELL order for", -best_bid_amount, "units at", best_bid)
                        orders.append(Order(product, best_bid, -best_bid_amount))
                else:
                    print("No breakout detected for", product)
            else:
                print("Not enough history for", product)
            
            result[product] = orders
        
        # Save updated history to traderData for persistence
        traderData = jsonpickle.encode(history)
        return result, conversions, traderData

    def get_best_ask(self, order_depth: OrderDepth):
        """Return the lowest sell price from the order depth, if available."""
        if order_depth.sell_orders:
            return min(order_depth.sell_orders.keys())
        return None

    def get_best_bid(self, order_depth: OrderDepth):
        """Return the highest buy price from the order depth, if available."""
        if order_depth.buy_orders:
            return max(order_depth.buy_orders.keys())
        return None
