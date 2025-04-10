# WHEN SUBMITTING FINAL SUBMISSION, CHANGE json AND ASSOCIATED FUNCTIONS TO jsonpickle AND REMOVE LOGGER

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List
import json
import math
import numpy as np

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
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

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()

class Trader:
    # resin strategy - stable at 10000
    def resin(self, state: TradingState, limit: int):
        return []

    # kelp strategy
    def kelp(self, state: TradingState, limit: int):
        return []

    def ink_improved(self, state: TradingState, limit: int, stored_data: dict = {}):    
        orders: List[Order] = []

        # 1. Retrieve order book data
        order_depth = state.order_depths["SQUID_INK"]
        position = state.position.get("SQUID_INK", 0)

        window_size = 3  # parameter to tweak
        volatility_factor = 4  # parameter to tweak



        if len(order_depth.buy_orders) == 0 or len(order_depth.sell_orders) == 0:
            return orders
        # 3. Compute current "fair value" as the mid of the best bid/ask
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        mid_price = (best_ask + best_bid) / 2

        stored_data["SQUID_INK"].append(mid_price)


        rolling_mean, rolling_std = self.get_rolling_metrics(
            stored_data["SQUID_INK"], window_size
        )

        # If not enough data to compute rolling metrics, fallback
        if rolling_mean is None or rolling_std is None:
            return orders  # or fallback to basic approach
        
        quote_distance = volatility_factor * rolling_std
        
        # 5. If the price is a "spike", consider contrarian trade
        z_score = (mid_price - rolling_mean) / (rolling_std if rolling_std != 0 else 1e-9)

        # Example mean reversion logic
        # (only do this if you have room in your inventory to take on a position)
        if abs(z_score) > volatility_factor and abs(position) < 0.9 * limit:
            # If price is well above the mean => short
            if z_score > volatility_factor:
                # Sell a small fraction
                quantity = min(5, limit - position)
                best_bid = max(order_depth.buy_orders.keys())
                # Use best_bid or a tick lower to get filled
                orders.append(Order("SQUID_INK", best_bid, -quantity))
            # If price is well below the mean => buy
            elif z_score < -1.5:
                quantity = min(5, limit + position)
                best_ask = min(order_depth.sell_orders.keys())
                orders.append(Order("SQUID_INK", best_ask, quantity))

        # 6. Market Making: place bid/ask around rolling_mean by quote_distance
        #    But also ensure we don't exceed our limit if we get filled
        desired_bid = rolling_mean - quote_distance
        desired_ask = rolling_mean + quote_distance

        # Round or truncate to integers if needed
        bid_price = round(desired_bid)
        ask_price = round(desired_ask)

        # Inventory management: if position is close to +limit, we might not place any new buys
        # if position is close to -limit, we might not place any new sells
        buy_quantity = max(0, limit - position)
        sell_quantity = max(0, limit + position)

        if buy_quantity > 0:
            orders.append(Order("SQUID_INK", bid_price, buy_quantity))

        if sell_quantity > 0:
            orders.append(Order("SQUID_INK", ask_price, -sell_quantity))

        return orders

    def get_rolling_metrics(self, historical_prices: List[float], window_size: int):
        if len(historical_prices) < window_size:
            return None, None
        window_data = historical_prices[-window_size:]
        r_mean = np.mean(window_data)
        r_std = np.std(window_data)
        return r_mean, r_std


    def run(self, state: TradingState):
        stored_data = json.loads(state.traderData) if state.traderData else {}

        for product in ["RAINFOREST_RESIN", "KELP", "SQUID_INK"]:
            if product not in stored_data:
                stored_data[product] = []

        POSITION_LIMITS = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "INK": 50
        }

        result = {}

        for product in ["RAINFOREST_RESIN", "KELP", "INK"]:
            if product == "RAINFOREST_RESIN":
                result["RAINFOREST_RESIN"] = self.resin(state, POSITION_LIMITS["RAINFOREST_RESIN"])
            elif product == "KELP":
                result["KELP"] = self.kelp(state, POSITION_LIMITS["KELP"])
            elif product == "INK":
                result["SQUID_INK"] = self.ink_improved(state, POSITION_LIMITS["INK"], stored_data)

        trader_data = json.dumps(stored_data, separators = (",", ":"))

        conversions = 0
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data