# WHEN SUBMITTING FINAL SUBMISSION, CHANGE json AND ASSOCIATED FUNCTIONS TO jsonpickle AND REMOVE LOGGER

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List
import jsonpickle
import math
import numpy as np

class Logger:
    def __init__(self) -> None:
        self.logs = ""

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, List[Order]], conversions: int, trader_data: str) -> None:
        print(self.logs)
        self.logs = ""

logger = Logger()

class Trader:
    def __init__(self):
        self.price_history: Dict[str, List[float]] = {"SQUID_INK": []}
        self.window_size = 50

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
                    order_depth.sell_orders[best_ask] += quantity
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if best_bid >= fair_value + take_width:
                quantity = min(best_bid_amount, limit + position)
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_bid, -1 * quantity))
                    sell_order_volume += quantity
                    order_depth.buy_orders[best_bid] -= quantity
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]

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
        disregard_edge = 0

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

    # kelp strategy
    def kelp(self, state: TradingState, limit: int):
        orders: List[Order] = []

        order_depth = state.order_depths["KELP"]
        position = state.position["KELP"] if "KELP" in state.position else 0

        # fair value calculation: it seems that the fair is the mid-price of the highest ask and lowest bid
        mm_ask = max(order_depth.sell_orders.keys())
        mm_bid = min(order_depth.buy_orders.keys())

        fair_value = (mm_ask + mm_bid) / 2
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
                    orders.append(Order("KELP", best_ask, quantity))
                    buy_order_volume += quantity
                    order_depth.sell_orders[best_ask] += quantity
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if best_bid >= fair_value + take_width:
                quantity = min(best_bid_amount, limit + position)
                if quantity > 0:
                    orders.append(Order("KELP", best_bid, -1 * quantity))
                    sell_order_volume += quantity
                    order_depth.buy_orders[best_bid] -= quantity
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]

        # make 0 ev trades to try to get back to 0 position
        orders, buy_order_volume, sell_order_volume = self.clear_orders(
            state,
            "KELP",
            buy_order_volume,
            sell_order_volume,
            fair_value,
            take_width,
            limit,
            orders
        )

        # market making
        # if prices are at most this much above/below the fair, do not market make
        disregard_edge = 0

        # if prices are at most this much above/below the fair, join (market make at the same price)
        join_edge = 1
        asks_above_fair = [price for price in order_depth.sell_orders.keys() if price > fair_value + disregard_edge]
        bids_below_fair = [price for price in order_depth.buy_orders.keys() if price < fair_value - disregard_edge]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = fair_value + 1
        if best_ask_above_fair != None:
            # joining criteria
            if best_ask_above_fair - fair_value <= join_edge:
                ask = best_ask_above_fair
            # pennying criteria (undercutting by the minimum)
            else:
                ask = best_ask_above_fair - 1

        bid = fair_value - 1
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        # how many buy orders we could put out
        buy_quantity = limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order("KELP", round(bid), buy_quantity))

        sell_quantity = limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order("KELP", round(ask), -1 * sell_quantity))

        return orders
    
    #squid ink
    def squid_ink(self, state: TradingState, product: str, limit: int) -> List[Order]:
        orders = []
        order_depth = state.order_depths.get(product, None)
        if order_depth is None:
            return []

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        if best_bid is None or best_ask is None:
            return []

        mid_price = (best_bid + best_ask) / 2
        self.price_history[product].append(mid_price)

        if len(self.price_history[product]) > self.window_size:
            self.price_history[product] = self.price_history[product][-self.window_size:]

        forecast_price = None
        if len(self.price_history[product]) >= self.window_size:
            try:
                prices = np.array(self.price_history[product])
                diffs = np.diff(prices)
                y = diffs[1:]
                X = diffs[:-1]
                if len(X) >= 2:
                    phi = np.dot(X, y) / np.dot(X, X)
                    delta_forecast = phi * diffs[-1]
                    forecast_price = prices[-1] + delta_forecast
            except:
                forecast_price = None

        if forecast_price is not None:
            position = state.position.get(product, 0)

            if forecast_price > mid_price and position < limit:
                buy_qty = min(limit - position, order_depth.sell_orders.get(best_ask, 1))
                orders.append(Order(product, best_ask, buy_qty))

            elif forecast_price < mid_price and position > -limit:
                sell_qty = min(position + limit, order_depth.buy_orders.get(best_bid, 1))
                orders.append(Order(product, best_bid, -sell_qty))

        return orders
    
    def run(self, state: TradingState):
        old_trader_data = jsonpickle.decode(state.traderData) if state.traderData else {}
        new_trader_data = {}

        POSITION_LIMITS = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50
        }

        result = {}

        for product in ["RAINFOREST_RESIN", "KELP", "SQUID_INK"]:
            if product == "RAINFOREST_RESIN":
                result["RAINFOREST_RESIN"] = self.resin(state, POSITION_LIMITS["RAINFOREST_RESIN"])
            elif product == "KELP":
                result["KELP"] = self.kelp(state, POSITION_LIMITS["KELP"])
            elif product == "SQUID_INK":
                result["SQUID_INK"] = self.squid_ink(state, "SQUID_INK", POSITION_LIMITS["SQUID_INK"])

        trader_data = jsonpickle.encode(new_trader_data, separators=(",", ":"))
        conversions = 0
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
