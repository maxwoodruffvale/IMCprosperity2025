import jsonpickle
from datamodel import OrderDepth, TradingState, Order, Trade
from typing import List, Dict

class Trader:
    FAST_WINDOW = 3   # Fast moving average window size
    SLOW_WINDOW = 7   # Slow moving average window size

    def run(self, state: TradingState):
        # Deserialize traderData to restore our historical prices per product.
        # Expecting traderData to be a JSON-encoded dictionary mapping product -> list of prices.
        try:
            history = jsonpickle.decode(state.traderData)
            if not isinstance(history, dict):
                history = {}
        except Exception:
            history = {}

        print("traderData:", state.traderData)
        print("Observations:", state.observations)

        result: Dict[str, List[Order]] = {}
        
        # Process each product available in order_depths.
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            
            # Initialize history list for product if not present.
            if product not in history:
                history[product] = []

            # Update historical prices from market trades.
            if product in state.market_trades:
                for trade in state.market_trades[product]:
                    history[product].append(trade.price)

            # Only keep the most recent SLOW_WINDOW prices.
            if len(history[product]) > self.SLOW_WINDOW:
                history[product] = history[product][-self.SLOW_WINDOW:]
            
            # If we have enough data points, compute moving averages.
            if len(history[product]) >= self.SLOW_WINDOW:
                fast_prices = history[product][-self.FAST_WINDOW:]
                slow_prices = history[product]
                fast_ma = sum(fast_prices) / len(fast_prices)
                slow_ma = sum(slow_prices) / len(slow_prices)
                acceptable_price = fast_ma  # Using fast MA as our reference price.
                
                print(f"Product: {product} | Fast MA: {fast_ma:.2f} | Slow MA: {slow_ma:.2f} | Acceptable Price: {acceptable_price:.2f}")
                
                # If fast MA > slow MA, bullish signal -> look to buy.
                if fast_ma > slow_ma and len(order_depth.sell_orders) > 0:
                    # For sell orders, best ask is the lowest price.
                    best_ask = self.get_best_ask(order_depth)
                    if best_ask is not None:
                        best_ask_amount = order_depth.sell_orders[best_ask]
                        # Since sell order volumes are negative, -best_ask_amount gives a positive buy quantity.
                        if best_ask < acceptable_price:
                            print("BUY", str(-best_ask_amount) + "x", best_ask)
                            orders.append(Order(product, best_ask, -best_ask_amount))
                
                # If fast MA < slow MA, bearish signal -> look to sell.
                elif fast_ma < slow_ma and len(order_depth.buy_orders) > 0:
                    # For buy orders, best bid is the highest price.
                    best_bid = self.get_best_bid(order_depth)
                    if best_bid is not None:
                        best_bid_amount = order_depth.buy_orders[best_bid]
                        # Buy orders have positive volume so -best_bid_amount yields a negative (sell) quantity.
                        if best_bid > acceptable_price:
                            print("SELL", str(best_bid_amount) + "x", best_bid)
                            orders.append(Order(product, best_bid, -best_bid_amount))
            else:
                print(f"Not enough historical data for {product}. Collected {len(history[product])} prices.")
            
            result[product] = orders

        # No conversion logic is implemented, so we return 0 for conversions.
        conversions = 0
        
        # Serialize the updated history dictionary back into a string for traderData.
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