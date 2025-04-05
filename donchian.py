from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List, Dict
import json
import math
import statistics

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
        if not state.traderData:
            stored_data = {
                "ask_history": {
                    "RAINFOREST_RESIN": [],
                    "KELP": []
                },
                "bid_history": {
                    "RAINFOREST_RESIN": [],
                    "KELP": []
                }
            }
        else:
            stored_data = json.loads(state.traderData)

        ask_history = stored_data["ask_history"]
        bid_history = stored_data["bid_history"]

        DONCHIAN_WINDOW = 20

        TRADE_SIZE = 20

        position_limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50
        }

        order_depths: Dict[str, OrderDepth] = state.order_depths
        my_positions: Dict[str, int] = state.position

        result = {}

        for product in ["RAINFOREST_RESIN", "KELP"]:
            if product not in order_depths:
                continue

            limit = position_limits[product]
            current_pos = my_positions.get(product, 0)
            depth = order_depths[product]

            best_ask = min(depth.sell_orders.keys())
            best_bid = max(depth.buy_orders.keys())

            ask_history[product].append(best_ask)
            bid_history[product].append(best_bid)

            ask_history[product] = ask_history[product][-DONCHIAN_WINDOW:]
            bid_history[product] = bid_history[product][-DONCHIAN_WINDOW:]

            highest_ask = max(ask_history[product])
            lowest_bid = min(ask_history[product])

            mid_price = (highest_ask + lowest_bid) / 2

            orders: List[Order] = []

            max_buy_qty = limit - current_pos
            max_sell_qty = current_pos + limit

            if best_ask > highest_ask:
                if max_buy_qty > 0:
                    buy_qty = min(max_buy_qty, TRADE_SIZE)
                    orders.append(Order(product, best_ask, buy_qty))

            if best_bid < lowest_bid:
                if max_sell_qty > 0:
                    sell_qty = min(max_sell_qty, TRADE_SIZE)
                    orders.append(Order(product, best_bid, -sell_qty))

            result[product] = orders

        new_trader_data_dict = {
            "ask_history": ask_history,
            "bid_history": bid_history
        }
        new_trader_data = json.dumps(new_trader_data_dict)
        conversions = 0
        logger.flush(state, result, conversions, new_trader_data)
        return result, conversions, new_trader_data