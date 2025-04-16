# import math
# import statistics
# import numpy as np
# import json
# import pandas as pd
# from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
# from typing import Any, List, Dict

# class Logger:
#     def __init__(self) -> None:
#         self.logs = ""
#         self.max_log_length = 3750

#     def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
#         self.logs += sep.join(map(str, objects)) + end

#     def flush(self, state: TradingState, orders: Dict[Symbol, List[Order]], conversions: int, trader_data: str) -> None:
#         base_length = len(
#             self.to_json([
#                 self.compress_state(state, ""),
#                 self.compress_orders(orders),
#                 conversions,
#                 "",
#                 "",
#             ])
#         )

#         max_item_length = (self.max_log_length - base_length) // 3

#         print(
#             self.to_json([
#                 self.compress_state(state, self.truncate(state.traderData, max_item_length)),
#                 self.compress_orders(orders),
#                 conversions,
#                 self.truncate(trader_data, max_item_length),
#                 self.truncate(self.logs, max_item_length),
#             ])
#         )

#         self.logs = ""

#     def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
#         return [
#             state.timestamp,
#             trader_data,
#             self.compress_listings(state.listings),
#             self.compress_order_depths(state.order_depths),
#             self.compress_trades(state.own_trades),
#             self.compress_trades(state.market_trades),
#             state.position,
#             self.compress_observations(state.observations),
#         ]

#     def compress_listings(self, listings: Dict[Symbol, Listing]) -> List[List[Any]]:
#         return [[listing.symbol, listing.product, listing.denomination] for listing in listings.values()]

#     def compress_order_depths(self, order_depths: Dict[Symbol, OrderDepth]) -> Dict[Symbol, List[Any]]:
#         return {symbol: [depth.buy_orders, depth.sell_orders] for symbol, depth in order_depths.items()}

#     def compress_trades(self, trades: Dict[Symbol, List[Trade]]) -> List[List[Any]]:
#         return [
#             [trade.symbol, trade.price, trade.quantity, trade.buyer, trade.seller, trade.timestamp]
#             for arr in trades.values() for trade in arr
#         ]

#     def compress_observations(self, observations: Observation) -> List[Any]:
#         conversion_observations = {
#             product: [
#                 obs.bidPrice, obs.askPrice, obs.transportFees,
#                 obs.exportTariff, obs.importTariff, obs.sugarPrice, obs.sunlightIndex
#             ]
#             for product, obs in observations.conversionObservations.items()
#         }
#         return [observations.plainValueObservations, conversion_observations]

#     def compress_orders(self, orders: Dict[Symbol, List[Order]]) -> List[List[Any]]:
#         return [[order.symbol, order.price, order.quantity] for arr in orders.values() for order in arr]

#     def to_json(self, value: Any) -> str:
#         return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

#     def truncate(self, value: str, max_length: int) -> str:
#         return value if len(value) <= max_length else value[:max_length - 3] + "..."

# logger = Logger()

# class Trader:
#     def __init__(self):
#         self.recent_rock_prices = []
#         self.recent_voucher_prices = []

#     def volcanic_rock_strategy(self, state: TradingState) -> List[Order]:
#         orders = []
#         rock_depth = state.order_depths.get("VOLCANIC_ROCK")
#         voucher_depth = state.order_depths.get("VOLCANIC_ROCK_VOUCHER_10000")
#         rock_position = state.position["VOLCANIC_ROCK"] if "VOLCANIC_ROCK" in state.position else 0
#         coupon_position = state.position["VOLCANIC_ROCK_VOUCHER_10000"] if "VOLCANIC_ROCK_VOUCHER_10000" in state.position else 0
#         if rock_depth and rock_depth.buy_orders and rock_depth.sell_orders and \
#            voucher_depth and voucher_depth.buy_orders and voucher_depth.sell_orders:

#             rock_mid = 0.5 * (max(rock_depth.buy_orders.keys()) + min(rock_depth.sell_orders.keys()))
#             voucher_mid = 0.5 * (max(voucher_depth.buy_orders.keys()) + min(voucher_depth.sell_orders.keys()))

#             self.recent_rock_prices.append(rock_mid)
#             self.recent_voucher_prices.append(voucher_mid)

#             if len(self.recent_rock_prices) >= 50:
#                 rock_arr = np.array(self.recent_rock_prices[-50:])
#                 voucher_arr = np.array(self.recent_voucher_prices[-50:])
#                 spread = rock_arr - voucher_arr
#                 zscore = (spread[-1] - np.mean(spread)) / np.std(spread)

#                 QTY = 5
#                 if zscore > 1.5:
#                     # Rock is expensive, short rock buy voucher
#                     rock_bid = max(rock_depth.buy_orders.keys())
#                     #max_bid_amount = rock_depth.buy_orders[rock_bid]
#                     voucher_ask = min(voucher_depth.sell_orders.keys())
#                     #max_ask_amount = voucher_depth.sell_orders[voucher_ask]
#                     orders.append(Order("VOLCANIC_ROCK", rock_bid, max(-400-rock_position,-QTY)))
#                     orders.append(Order("VOLCANIC_ROCK_VOUCHER_10000", voucher_ask, min(200-coupon_position,QTY)))
#                 elif zscore < -1.5:
#                     # Rock is cheap, buy rock sell voucher
#                     rock_ask = min(rock_depth.sell_orders.keys())
#                     #max_ask_amount = rock_depth.sell_orders[rock_ask]
#                     voucher_bid = max(voucher_depth.buy_orders.keys())
#                     #max_bid_amount = voucher_depth.buy_orders[voucher_bid]
#                     orders.append(Order("VOLCANIC_ROCK", rock_ask, min(400-rock_position,QTY)))
#                     orders.append(Order("VOLCANIC_ROCK_VOUCHER_10000", voucher_bid, max(-200-coupon_position,-QTY)))

#         return orders

#     def run(self, state: TradingState):
#         stored_data = json.loads(state.traderData) if state.traderData else {}
#         result = {"VOLCANIC_ROCK": self.volcanic_rock_strategy(state)}
#         trader_data = json.dumps(stored_data)
#         conversions = 0
#         logger.flush(state, result, conversions, trader_data)
#         return result, conversions, trader_data
import math
import statistics
import numpy as np
import json
import pandas as pd
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List, Dict

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: Dict[Symbol, List[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json([
                self.compress_state(state, ""),
                self.compress_orders(orders),
                conversions,
                "",
                "",
            ])
        )

        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json([
                self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                self.compress_orders(orders),
                conversions,
                self.truncate(trader_data, max_item_length),
                self.truncate(self.logs, max_item_length),
            ])
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: Dict[Symbol, Listing]) -> List[List[Any]]:
        return [[listing.symbol, listing.product, listing.denomination] for listing in listings.values()]

    def compress_order_depths(self, order_depths: Dict[Symbol, OrderDepth]) -> Dict[Symbol, List[Any]]:
        return {symbol: [depth.buy_orders, depth.sell_orders] for symbol, depth in order_depths.items()}

    def compress_trades(self, trades: Dict[Symbol, List[Trade]]) -> List[List[Any]]:
        return [
            [trade.symbol, trade.price, trade.quantity, trade.buyer, trade.seller, trade.timestamp]
            for arr in trades.values() for trade in arr
        ]

    def compress_observations(self, observations: Observation) -> List[Any]:
        conversion_observations = {
            product: [
                obs.bidPrice, obs.askPrice, obs.transportFees,
                obs.exportTariff, obs.importTariff, obs.sugarPrice, obs.sunlightIndex
            ]
            for product, obs in observations.conversionObservations.items()
        }
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: Dict[Symbol, List[Order]]) -> List[List[Any]]:
        return [[order.symbol, order.price, order.quantity] for arr in orders.values() for order in arr]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[:max_length - 3] + "..."

logger = Logger()

class Trader:
    def __init__(self):
        self.recent_rock_prices = []
        # Define the rolling window size for Bollinger Bands
        self.window_size = 30
        # Multiplier for standard deviation
        self.num_std = 1.5

    def volcanic_rock_strategy(self, state: TradingState) -> List[Order]:
        """
        A Bollinger Band mean-reversion strategy for VOLCANIC_ROCK.
        We'll:
          1) Compute the mid-price from best bid & ask.
          2) Keep a rolling window of mid-prices.
          3) Compute rolling mean & std.
          4) Go long if current mid < (mean - 2 std),
             go short if current mid > (mean + 2 std).
        """

        orders = []
        # Extract the order depth for Volcanic Rock
        rock_depth = state.order_depths.get("VOLCANIC_ROCK")

        # Current position in Volcanic Rock (default to 0)
        rock_position = state.position.get("VOLCANIC_ROCK", 0)

        # If there is enough market data
        if rock_depth and rock_depth.buy_orders and rock_depth.sell_orders:
            best_bid = max(rock_depth.buy_orders.keys())
            best_ask = min(rock_depth.sell_orders.keys())
            mid_price = 0.5 * (best_bid + best_ask)

            # Update our rolling list of mid-prices
            self.recent_rock_prices.append(mid_price)
            if len(self.recent_rock_prices) > self.window_size:
                self.recent_rock_prices.pop(0)

            # Only trade after we have enough data in the rolling window
            if len(self.recent_rock_prices) == self.window_size:
                mean_price = np.mean(self.recent_rock_prices)
                std_price = np.std(self.recent_rock_prices)

                upper_band = mean_price + self.num_std * std_price
                lower_band = mean_price - self.num_std * std_price

                # Decide how many units we want to trade
                qty = 10

                # If mid price is above the upper band, short Volcanic Rock
                if mid_price > upper_band:
                    # We short by placing a sell order at the current best_bid (or best_ask).
                    # Safer to use best_bid if we're trying to sell quickly.
                    # Ensure we don't exceed a position of -400 (arbitrary example limit)
                    sell_qty = max(-400 - rock_position, -qty)
                    if sell_qty < 0:
                        orders.append(Order("VOLCANIC_ROCK", best_bid, sell_qty))

                # If mid price is below the lower band, go long Volcanic Rock
                elif mid_price < lower_band:
                    # We buy by placing a buy order at the current best_ask.
                    # Ensure we don't exceed a position of +400
                    buy_qty = min(400 - rock_position, qty)
                    if buy_qty > 0:
                        orders.append(Order("VOLCANIC_ROCK", best_ask, buy_qty))

        return orders

    def run(self, state: TradingState):
        # Load any stored data (not used heavily here, but kept for completeness)
        stored_data = json.loads(state.traderData) if state.traderData else {}

        # Here, we generate orders only for Volcanic Rock using our Bollinger Band strategy
        volcanic_rock_orders = self.volcanic_rock_strategy(state)

        result = {
            "VOLCANIC_ROCK": volcanic_rock_orders
            # No other symbols traded in this example
        }

        # Convert stored_data back to string if needed
        trader_data = json.dumps(stored_data)

        # We do not handle conversions in this strategy
        conversions = 0

        # Flush logs
        logger.flush(state, result, conversions, trader_data)

        return result, conversions, trader_data
