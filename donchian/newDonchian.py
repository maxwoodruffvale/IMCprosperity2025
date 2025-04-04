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
            
            # Determine current price: prefer the last market trade price
            current_price = None
            if product in state.market_trades and state.market_trades[product]:
                current_price = state.market_trades[product][-1].price
            else:
                # If no market trades, try to derive a mid-price from order depth
                best_bid = self.get_best_bid(order_depth)
                best_ask = self.get_best_ask(order_depth)
                if best_bid is not None and best_ask is not None:
                    current_price = (best_bid + best_ask) / 2
            
            if current_price is None:
                result[product] = orders
                continue
            
            # Initialize or update price history for the product
            if product not in history:
                history[product] = []
            history[product].append(current_price)
            if len(history[product]) > LOOKBACK_PERIOD:
                history[product].pop(0)
            
            print("Product:", product)
            print("Current Price:", current_price)
            print("Price History (len={}):".format(len(history[product])), history[product])
            
            # Only attempt to trade if we have enough history to compute the channel
            if len(history[product]) == LOOKBACK_PERIOD:
                channel_high = max(history[product])
                channel_low = min(history[product])
                print("Channel High:", channel_high, "Channel Low:", channel_low)
                
                # Trigger a buy if current price is greater than or equal to channel high
                if current_price >= channel_high:
                    if order_depth.sell_orders:
                        best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                        print("Breakout UP! Placing BUY order for", -best_ask_amount, "units at", best_ask)
                        # Format order exactly like the sample (invert volume from sell orders)
                        orders.append(Order(product, best_ask, -best_ask_amount))
                # Trigger a sell if current price is less than or equal to channel low
                elif current_price <= channel_low:
                    if order_depth.buy_orders:
                        best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                        print("Breakout DOWN! Placing SELL order for", -best_bid_amount, "units at", best_bid)
                        orders.append(Order(product, best_bid, -best_bid_amount))
                else:
                    print("No breakout detected for", product)
            else:
                print("Not enough history for", product)
            
            result[product] = orders
        
        # Save the updated history into traderData
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
