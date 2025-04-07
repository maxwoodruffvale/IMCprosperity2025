from abc import abstractmethod
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
    # make 0ev trades to try to get back to 0 position
    def clear_orders(self,
                     state: TradingState,
                     product: str,
                     buy_order_volume,
                     sell_order_volume,
                     fair_value,
                     take_width,
                     limit,
                     orders: List[Order],
                     ):
        position = state.position[product] if product in state.position else 0

        order_depth = state.order_depths[product]

        position_after_take = position + buy_order_volume - sell_order_volume

        fair_for_bid = round(fair_value - take_width)
        fair_for_ask = round(fair_value + take_width)

        # how much we want to clear from our position after taking
        buy_quantity = limit - (position + buy_order_volume)
        sell_quantity = limit + (position - sell_order_volume)

        if position_after_take > 0:
            # total volume from all buy orders with price >= fair_for_ask
            clear_quantity = sum(volume for price, volume in order_depth.buy_orders.items() if price >= fair_for_ask)

            # if cq > position after take, then we only want to get to position 0
            # if cq < position after take, then we can only clear cq
            clear_quantity = min(clear_quantity, position_after_take)

            # if cq > more than how much we want to clear, don't. If we cant cq as much as we want, we can't
            sent_quantity = min(sell_quantity, clear_quantity)

            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -sent_quantity))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            clear_quantity = sum(abs(volume) for price, volume in order_depth.sell_orders.items() if price <= fair_for_bid)
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, sent_quantity))
                buy_order_volume += abs(sent_quantity)

        return orders, buy_order_volume, sell_order_volume

    # resin strategy
    def resin(self, state: TradingState, limit: int):
        orders: List[Order] = []

        order_depth = state.order_depths["RAINFOREST_RESIN"]
        position = state.position["RAINFOREST_RESIN"] if "RAINFOREST_RESIN" in state.position else 0

        fair_value = 10000
        take_width = 1

        buy_order_volume = 0
        sell_order_volume = 0

        if len(order_depth.sell_orders) != 0:
            # market taking
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if best_ask <= fair_value - take_width:
                quantity = min(best_ask_amount, limit - position)
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_ask, quantity))
                    buy_order_volume += quantity
                    # order_depth.sell_orders[best_ask] += quantity
                    # if order_depth.sell_orders[best_ask] == 0:
                    #     del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if best_bid >= fair_value + take_width:
                quantity = min(best_bid_amount, limit + position)
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_bid, -1 * quantity))
                    sell_order_volume += quantity
                    # order_depth.buy_orders[best_bid] -= quantity
                    # if order_depth.buy_orders[best_bid] == 0:
                    #     del order_depth.buy_orders[best_bid]

        # make 0 ev trades to try to get back to 0 position
        orders, buy_order_volume, sell_order_volume = self.clear_orders(
            state,
            "RAINFOREST_RESIN",
            buy_order_volume,
            sell_order_volume,
            fair_value,
            take_width,
            limit,
            orders
        )

        # market making
        # if prices are at most this much above/below the fair, do not market make
        disregard_edge = 1

        # if prices are at most this much above/below the fair, join (market make at the same price)
        join_edge = 1
        asks_above_fair = [price for price in order_depth.sell_orders.keys() if price > fair_value + disregard_edge]
        bids_below_fair = [price for price in order_depth.buy_orders.keys() if price < fair_value - disregard_edge]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = 10005
        if best_ask_above_fair != None:
            # joining criteria
            if best_ask_above_fair - fair_value <= join_edge:
                ask = best_ask_above_fair
            # pennying criteria (undercutting by the minimum)
            else:
                ask = best_ask_above_fair - 1

        bid = 9995
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        # how many buy orders we could put out
        buy_quantity = limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", round(bid), buy_quantity))

        sell_quantity = limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", round(ask), -1 * sell_quantity))

        return orders

    def kelp(self, state: TradingState, limit: int):
        orders: List[Order] = []

        return []

    def run(self, state: TradingState):
        old_trader_data = json.loads(state.traderData) if state.traderData else {}
        new_trader_data = {}

        POSITION_LIMITS = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50
        }

        result = {}

        for product in ["RAINFOREST_RESIN", "KELP"]:
            if product == "RAINFOREST_RESIN":
                result["RAINFOREST_RESIN"] = self.resin(state, POSITION_LIMITS["RAINFOREST_RESIN"])
            elif product == "KELP":
                result["KELP"] = self.kelp(state, POSITION_LIMITS["KELP"])

        trader_data = json.dumps(new_trader_data, separators = (",", ":"))

        conversions = 0
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data