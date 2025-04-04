from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import jsonpickle

class Trader:
    # Safe bounds for net position per product.
    SAFE_POSITION_LIMIT = 10  
    # Order size is fixed small size to minimize risk.
    ORDER_SIZE = 5  

    def run(self, state: TradingState):
        # Load persistent state if needed; here we simply pass it along.
        traderData = state.traderData if state.traderData else "SAFE_STRATEGY"
        result: Dict[str, List[Order]] = {}

        for product, order_depth in state.order_depths.items():
            orders: List[Order] = []
            current_position = state.position.get(product, 0)

            # Determine best bid and best ask from the order book.
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

            # Require both sides to calculate a meaningful mid-price.
            if best_bid is None or best_ask is None:
                result[product] = orders
                continue

            mid_price = (best_bid + best_ask) // 2

            # Set delta based on product volatility.
            delta = 2 if product.lower() == "kelp" else 1
            buy_price = mid_price - delta
            sell_price = mid_price + delta

            # Only place a buy order if the current position is negative or neutral.
            if current_position <= 0 and current_position < self.SAFE_POSITION_LIMIT:
                buy_quantity = self.ORDER_SIZE
                # Adjust quantity so as not to exceed the safe limit.
                if current_position + buy_quantity > self.SAFE_POSITION_LIMIT:
                    buy_quantity = self.SAFE_POSITION_LIMIT - current_position
                if buy_quantity > 0:
                    orders.append(Order(product, buy_price, buy_quantity))

            # Only place a sell order if the current position is positive or neutral.
            if current_position >= 0 and current_position > -self.SAFE_POSITION_LIMIT:
                sell_quantity = self.ORDER_SIZE
                # Adjust quantity so as not to exceed the safe limit.
                if current_position - sell_quantity < -self.SAFE_POSITION_LIMIT:
                    sell_quantity = current_position + self.SAFE_POSITION_LIMIT
                if sell_quantity > 0:
                    # For sell orders, the quantity is negative.
                    orders.append(Order(product, sell_price, -sell_quantity))

            result[product] = orders

        # No conversion requests in this simple strategy.
        conversions = 0

        # Persist state using traderData.
        traderData = traderData

        return result, conversions, traderData
