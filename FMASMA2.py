import jsonpickle
from datamodel import OrderDepth, TradingState, Order, Trade
from typing import List, Dict

class Trader:
    FAST_WINDOW = 3   # Fast moving average window size
    SLOW_WINDOW = 7   # Slow moving average window size
    TARGET_POSITION = 10  # Fixed target position for bullish signal

    def run(self, state: TradingState):
        # Restore historical price data from traderData, or initialize if not present.
        if state.traderData:
            try:
                history = jsonpickle.decode(state.traderData)
            except Exception as e:
                print("Error decoding traderData, starting fresh:", e)
                history = {}
        else:
            history = {}

        print("traderData:", state.traderData)
        print("Observations:", state.observations)

        result: Dict[str, List[Order]] = {}

        # Process each product in the order book.
        for product in state.order_depths: 
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            
            # Initialize history list for the product if needed.
            if product not in history:
                history[product] = []

            # Update historical prices from market trades.
            if product in state.market_trades:
                for trade in state.market_trades[product]:
                    history[product].append(trade.price)
            
            # Keep only the most recent SLOW_WINDOW prices.
            if len(history[product]) > self.SLOW_WINDOW:
                history[product] = history[product][-self.SLOW_WINDOW:]
            
            # Only act if we have enough data.
            if len(history[product]) >= self.SLOW_WINDOW:
                # Compute moving averages.
                fast_prices = history[product][-self.FAST_WINDOW:]
                slow_prices = history[product]
                fast_ma = sum(fast_prices) / len(fast_prices)
                slow_ma = sum(slow_prices) / len(slow_prices)
                acceptable_price = fast_ma  # Using FMA as the reference price
                
                print(f"Product: {product} | Fast MA: {fast_ma:.2f} | Slow MA: {slow_ma:.2f} | Acceptable Price: {acceptable_price:.2f}")

                # Determine the target position based on the signal.
                if fast_ma > slow_ma:
                    target_position = self.TARGET_POSITION
                elif fast_ma < slow_ma:
                    target_position = -self.TARGET_POSITION
                else:
                    target_position = 0

                current_position = state.position.get(product, 0)
                delta = target_position - current_position

                # If we already are at the target position, do nothing.
                if delta == 0:
                    print(f"No action needed for {product}. Current position {current_position} matches target.")
                # Need to buy: delta > 0 means we want a more positive (long) position.
                elif delta > 0:
                    if order_depth.sell_orders:
                        best_ask = min(order_depth.sell_orders.keys())
                        available_volume = -order_depth.sell_orders[best_ask]  # Convert negative volume to positive
                        order_volume = min(delta, available_volume)
                        # Only buy if the ask price is attractive.
                        if best_ask < acceptable_price:
                            print(f"BUY {order_volume}x {best_ask} for {product}")
                            orders.append(Order(product, best_ask, order_volume))
                        else:
                            print(f"Not buying {product}: best ask {best_ask} not below acceptable price {acceptable_price:.2f}")
                    else:
                        print(f"No sell orders available for {product} to buy from.")
                # Need to sell: delta < 0 means we want a more negative (short) position.
                elif delta < 0:
                    if order_depth.buy_orders:
                        best_bid = max(order_depth.buy_orders.keys())
                        available_volume = order_depth.buy_orders[best_bid]
                        order_volume = min(-delta, available_volume)
                        # Only sell if the bid price is attractive.
                        if best_bid > acceptable_price:
                            print(f"SELL {order_volume}x {best_bid} for {product}")
                            orders.append(Order(product, best_bid, -order_volume))
                        else:
                            print(f"Not selling {product}: best bid {best_bid} not above acceptable price {acceptable_price:.2f}")
                    else:
                        print(f"No buy orders available for {product} to sell into.")
            else:
                print(f"Not enough historical data for {product}. Collected {len(history[product])} prices.")
            
            result[product] = orders

        conversions = 0  # No conversion logic for now.
        traderData = jsonpickle.encode(history)
        return result, conversions, traderData
