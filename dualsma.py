from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List
import json
import math

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
    def run(self, state: TradingState):
        try:
            stored_data = json.loads(state.traderData) if state.traderData else {}
        except:
            stored_data = {}

        for product in ["RAINFOREST_RESIN", "KELP"]:
            if product not in stored_data:
                stored_data[product] = {}
            if "prices" not in stored_data[product]:
                stored_data[product]["prices"] = []

        SHORT_WINDOW = 5
        LONG_WINDOW = 15
        MAX_BASE_TRADE_SIZE = 50
        POSITION_LIMIT = 50

        def compute_mid_price(order_depth: OrderDepth) -> float:
            if len(order_depth.buy_orders) == 0 or len(order_depth.sell_orders) == 0:
                return None
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2.0

        def simple_moving_average(data: List[float], window: int) -> float:
            if len(data) < window:
                return None
            return sum(data[-window:]) / window

        result = {}

        for product in ["RAINFOREST_RESIN", "KELP"]:
            if product not in state.order_depths:
                continue
            order_depth: OrderDepth = state.order_depths[product]
            current_position = state.position.get(product, 0)

            mid_price = compute_mid_price(order_depth)
            if mid_price is None:
                result[product] = []
                continue

            stored_data[product]["prices"].append(mid_price)

            short_sma = simple_moving_average(stored_data[product]["prices"], SHORT_WINDOW)
            long_sma = simple_moving_average(stored_data[product]["prices"], LONG_WINDOW)

            if short_sma is None or long_sma is None:
                result[product] = []
                continue

            sma_diff = short_sma - long_sma

            orders: List[Order] = []

            relative_diff = abs(sma_diff) / long_sma if long_sma != 0 else 0
            scale = min(1 + relative_diff, 2)
            trade_size = round(MAX_BASE_TRADE_SIZE * scale)

            max_buyable = POSITION_LIMIT - current_position
            max_sellable = current_position + POSITION_LIMIT

            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())

            if product == "RAINFOREST_RESIN":
                if best_ask < short_sma:
                    qty = min(trade_size, max_buyable)
                    orders.append(Order(product, best_ask, qty))
                elif best_bid > short_sma:
                    qty = min(trade_size, max_sellable)
                    orders.append(Order(product, best_bid, -qty))

            else:
                if sma_diff > 0:
                    qty = min(trade_size, max_buyable)
                    if qty > 0 and best_ask < short_sma:
                        orders.append(Order(product, best_ask, qty))

                elif sma_diff < 0:
                    qty = min(trade_size, max_sellable)
                    if qty > 0 and best_bid > short_sma:
                        orders.append(Order(product, best_bid, -qty))

            result[product] = orders

        new_trader_data = json.dumps(stored_data)
        conversions = 0
        logger.flush(state, result, conversions, new_trader_data)
        return result, conversions, new_trader_data