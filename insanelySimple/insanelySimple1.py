from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import jsonpickle

class Trader:
    # Safe bounds for net position per product. These values ensure we don’t build up too large a position.
    SAFE_POSITION_LIMIT = 10  
    # Order size is fixed small size to minimize risk.
    ORDER_SIZE = 5  

    def run(self, state: TradingState):
        # Load persistent state if needed; here we simply pass it along.
        traderData = state.traderData if state.traderData else "SAFE_STRATEGY"

        result: Dict[str, List[Order]] = {}

        # Process each product in the order depth
        for product, order_depth in state.order_depths.items():
            orders: List[Order] = []
            
            # Get current position; default to 0 if not in state
            current_position = state.position.get(product, 0)
            
            # Determine best bid and best ask from the order book.
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
            
            # We require both sides for our mid-price calculation; otherwise, skip trading for safety.
            if best_bid is None or best_ask is None:
                result[product] = orders
                continue

            # Compute the mid price
            mid_price = (best_bid + best_ask) // 2

            # Determine price adjustment delta based on product volatility.
            # Kelp is assumed more volatile; thus a wider spread is used.
            if product.lower() == "kelp":
                delta = 2
            else:
                delta = 1

            # Define our buy and sell prices based on the mid price and delta.
            buy_price = mid_price - delta
            sell_price = mid_price + delta

            # Place a buy order if our current position is below the safe limit.
            if current_position < self.SAFE_POSITION_LIMIT:
                # The order quantity is set to a small fixed amount.
                buy_quantity = self.ORDER_SIZE
                # Ensure that adding this buy order would not exceed the safe position limit.
                if current_position + buy_quantity > self.SAFE_POSITION_LIMIT:
                    buy_quantity = self.SAFE_POSITION_LIMIT - current_position
                if buy_quantity > 0:
                    orders.append(Order(product, buy_price, buy_quantity))
            
            # Place a sell order if our current position is above the negative safe limit.
            if current_position > -self.SAFE_POSITION_LIMIT:
                sell_quantity = self.ORDER_SIZE
                # Ensure that adding this sell order (which decreases our position) would not exceed the safe limit.
                if current_position - sell_quantity < -self.SAFE_POSITION_LIMIT:
                    sell_quantity = current_position + self.SAFE_POSITION_LIMIT
                if sell_quantity > 0:
                    # For sell orders, quantity is negative.
                    orders.append(Order(product, sell_price, -sell_quantity))
            
            result[product] = orders

        # For this safe strategy, we are not performing any conversions.
        conversions = 0

        # Save our state (here we just pass along a simple string; more complex state can be stored via jsonpickle)
        traderData = traderData

        return result, conversions, traderData
