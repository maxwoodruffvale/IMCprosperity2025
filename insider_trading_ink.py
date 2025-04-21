from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List, Dict, Optional
import numpy as np
import json
import math
from statistics import NormalDist
from collections import deque
from dataclasses import dataclass

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
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2

            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."

            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out


logger = Logger()

class Trader:
    # def __init__(self):
    #     self.returns_history = {v: deque(maxlen=PARAMS[v]["returns_history_window"]) for v in PARAMS.keys()}    
    def ink_strategy(self, state: TradingState, limit: int) -> List[Order]:
        orders: List[Order] = []

        order_depth = state.order_depths["SQUID_INK"]
        position = state.position["SQUID_INK"] if "SQUID_INK" in state.position else 0
        trades = state.market_trades.get("SQUID_INK", [])
        trades = [t for t in trades if t.timestamp == state.timestamp-100]
        if any(t.buyer == "Olivia" for t in trades):
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            quantity = min(best_ask_amount, limit - position)
            orders.append(Order("SQUID_INK", best_ask, quantity))
        if any(t.seller == "Olivia" for t in trades):
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            quantity = min(best_bid_amount, limit + position)
            orders.append(Order("SQUID_INK", best_bid, -1 * quantity))
        return orders
    def run(self, state: TradingState):
        stored_data = json.loads(state.traderData) if state.traderData else {}

        # for product in ["RAINFOREST_RESIN",
        #                 "KELP",
        #                 "SQUID_INK",
        #                 "CROISSANTS",
        #                 "JAMS",
        #                 "DJEMBES",
        #                 "PICNIC_BASKET1",
        #                 "PICNIC_BASKET2",
        #                 "VOLCANIC_ROCK",
        #                 "VOLCANIC_ROCK_VOUCHER_9500",
        #                 "VOLCANIC_ROCK_VOUCHER_9750",
        #                 "VOLCANIC_ROCK_VOUCHER_10000",
        #                 "VOLCANIC_ROCK_VOUCHER_10250",
        #                 "VOLCANIC_ROCK_VOUCHER_10500",
        #                 "MAGNIFICENT_MACARONS"
        #                 ]:
        #     if product not in stored_data:
        #         stored_data[product] = {
        #             "spread_history": [],
        #             "fair_value": 0,
        #         }

        POSITION_LIMITS = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBES": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
            "VOLCANIC_ROCK": 400,
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200,
            "MAGNIFICENT_MACARONS": 75,
        }

        result = {}

        for product in ["RAINFOREST_RESIN",
                        "KELP",
                        "SQUID_INK",
                        "CROISSANTS",
                        "JAMS",
                        "DJEMBES",
                        "PICNIC_BASKET1",
                        "PICNIC_BASKET2",
                        "VOLCANIC_ROCK",
                        "VOLCANIC_ROCK_VOUCHER_9500",
                        "VOLCANIC_ROCK_VOUCHER_9750",
                        "VOLCANIC_ROCK_VOUCHER_10000",
                        "VOLCANIC_ROCK_VOUCHER_10250",
                        "VOLCANIC_ROCK_VOUCHER_10500",
                        "MAGNIFICENT_MACARONS"
                        ]:
            # if product == "RAINFOREST_RESIN":
            #     result["RAINFOREST_RESIN"] = self.resin_strategy(state, POSITION_LIMITS["RAINFOREST_RESIN"])
            # if product == "KELP":
            #     result["KELP"] = self.kelp_strategy(state, POSITION_LIMITS["KELP"])
            if product == "SQUID_INK":
                result["SQUID_INK"] = self.ink_strategy(state, POSITION_LIMITS["SQUID_INK"])
            # if product == "PICNIC_BASKET1":
            #     arb1_exists = True
            #     arb2_exists = True
            #     arb3_exists = True
            #     picnic1_first = True
            #     picnic2_first = True
            #     djembes_first = True
            #     while arb1_exists or arb2_exists or arb3_exists:
            #         best = self.spread_picker(state,
            #                                   arb1_exists,
            #                                   arb2_exists,
            #                                   arb3_exists,
            #                                   stored_data,
            #                                   )
            #         if best == "PICNIC_BASKET1":
            #             orders, croissants_orders, jams_orders, djembes_orders = self.picnic1_strategy(state, POSITION_LIMITS["PICNIC_BASKET1"], stored_data, picnic1_first)
            #             result["PICNIC_BASKET1"] = orders
            #             result["CROISSANTS"] = croissants_orders
            #             result["JAMS"] = jams_orders
            #             result["DJEMBES"] = djembes_orders
            #             picnic1_first = False
            #             if orders == [] or orders[0].quantity == 0:
            #                 arb1_exists = False
            #         elif best == "PICNIC_BASKET2":
            #             orders, croissants_orders, jams_orders = self.picnic2_strategy(state, POSITION_LIMITS[
            #                 "PICNIC_BASKET2"], stored_data, picnic2_first)
            #             result["PICNIC_BASKET2"] = orders
            #             result["CROISSANTS"] = croissants_orders
            #             result["JAMS"] = jams_orders
            #             picnic2_first = False
            #             if orders == [] or orders[0].quantity == 0:
            #                 arb2_exists = False
            #         elif best == "DJEMBES":
            #             orders, picnic1_orders, picnic2_orders = self.djembes_strategy(state, POSITION_LIMITS[
            #                 "DJEMBES"], stored_data, djembes_first)
            #             result["DJEMBES"] = orders
            #             result["PICNIC_BASKET1"] = picnic1_orders
            #             result["PICNIC_BASKET2"] = picnic2_orders
            #             djembes_first = False
            #             if orders == [] or orders[0].quantity == 0:
            #                 arb3_exists = False
            #         else:
            #             break
            # if product == "VOLCANIC_ROCK":
            #     result["VOLCANIC_ROCK"] = self.rock_strategy(state,
            #                                                  POSITION_LIMITS["VOLCANIC_ROCK"],
            #                                                  stored_data,
            #                                                  )
            # if product == "VOLCANIC_ROCK_VOUCHER_9500":
            #     result["VOLCANIC_ROCK_VOUCHER_9500"] = self.voucher_strategy(state,
            #                                                                POSITION_LIMITS[
            #                                                                    "VOLCANIC_ROCK_VOUCHER_9500"],
            #                                                                stored_data,
            #                                                                 "VOLCANIC_ROCK_VOUCHER_9500",
            #                                                                  9500,
            #                                                                  100,
            #                                                                  1.4
            #                                                                )
            # if product == "VOLCANIC_ROCK_VOUCHER_9750":
            #     result["VOLCANIC_ROCK_VOUCHER_9750"] = self.voucher_strategy(state,
            #                                                                POSITION_LIMITS[
            #                                                                    "VOLCANIC_ROCK_VOUCHER_9750"],
            #                                                                stored_data,
            #                                                                  "VOLCANIC_ROCK_VOUCHER_9750",
            #                                                                  9750,
            #                                                                  100,
            #                                                                  0.6
            #                                                                )
            # if product == "VOLCANIC_ROCK_VOUCHER_10000":
            #     result["VOLCANIC_ROCK_VOUCHER_10000"] = self.voucher_strategy(state,
            #                                                                POSITION_LIMITS[
            #                                                                    "VOLCANIC_ROCK_VOUCHER_10000"],
            #                                                                stored_data,
            #                                                                   "VOLCANIC_ROCK_VOUCHER_10000",
            #                                                                   10000,
            #                                                                   100,
            #                                                                   0.25
            #                                                                )
            # if product == "VOLCANIC_ROCK_VOUCHER_9750":
            #     result["VOLCANIC_ROCK_VOUCHER_9750"] = self._trade_vouchers_v2("VOLCANIC_ROCK_VOUCHER_9750",
            #                                                                    state
            #                                                                    )
            # if product == "VOLCANIC_ROCK_VOUCHER_10000":
            #     result["VOLCANIC_ROCK_VOUCHER_10000"] = self._trade_vouchers_v2("VOLCANIC_ROCK_VOUCHER_10000",
            #                                                                state
            #                                                                )
            # if product == "VOLCANIC_ROCK_VOUCHER_10250":
            #     result["VOLCANIC_ROCK_VOUCHER_10250"] = self.voucher_hard_strategy(state,
            #                                                                POSITION_LIMITS[
            #                                                                    "VOLCANIC_ROCK_VOUCHER_10250"],
            #                                                                  "VOLCANIC_ROCK_VOUCHER_10250",
            #                                                                   4
            #                                                                )
            # if product == "VOLCANIC_ROCK_VOUCHER_10500":
            #     result["VOLCANIC_ROCK_VOUCHER_10500"] = self.voucher_hard_strategy(state,
            #                                                                POSITION_LIMITS[
            #                                                                    "VOLCANIC_ROCK_VOUCHER_10500"],
            #                                                                  "VOLCANIC_ROCK_VOUCHER_10500",
            #                                                                   2,
            #                                                                )
            # if product == "MAGNIFICENT_MACARONS":
            #     result["MAGNIFICENT_MACARONS"], conversions = self.compute_macaron_orders(state)

        trader_data = json.dumps(stored_data, separators=(",", ":"))

        # GET RID OF IN FINALL!!!!!!!!
        conversions = 0

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data