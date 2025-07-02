# WHEN SUBMITTING FINAL SUBMISSION, CHANGE json AND ASSOCIATED FUNCTIONS TO jsonpickle AND REMOVE LOGGER

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List, Dict
import numpy as np
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

        buy_order_volume = 0
        sell_order_volume = 0

        fair_value = 10000
        take_width = 1

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
        disregard_edge = 1

        # if prices are at most this much above/below the fair, join (market make at the same price)
        join_edge = 2
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
    # def kelp(self, state: TradingState, limit: int, stored_data):
    #     orders: List[Order] = []
        
    #     # Safety checks
    #     if "KELP" not in state.order_depths:
    #         return []
        
    #     order_depth = state.order_depths["KELP"]
    #     if not order_depth.sell_orders or not order_depth.buy_orders:
    #         return []

    #     # Improved fair value calculation using VWAP
    #     total_bid_volume = sum(order_depth.buy_orders.values())
    #     total_ask_volume = sum(abs(v) for v in order_depth.sell_orders.values())
    #     vwap_bid = sum(p * v for p, v in order_depth.buy_orders.items()) / total_bid_volume
    #     vwap_ask = sum(p * abs(v) for p, v in order_depth.sell_orders.items()) / total_ask_volume
    #     fair_value = (vwap_bid + vwap_ask) / 2

    #     # Dynamic position-based spread
    #     position = state.position.get("KELP", 0)
    #     position_factor = position / limit
    #     base_spread = 2
    #     dynamic_spread = base_spread * (1 + abs(position_factor))

    #     # Improved moving average calculation
    #     window_size = 5  # Reduced from 20 for faster reaction
    #     threshold_to_not_add = dynamic_spread * 2

    #     fair_history = stored_data["fair_history"]["KELP"]
    #     if len(fair_history) == 0:
    #         threshold_price = fair_value
    #     else:
    #         # Exponential moving average instead of simple MA
    #         alpha = 0.3  # Smoothing factor
    #         threshold_price = alpha * fair_value + (1 - alpha) * fair_history[-1]

    #     # Update history with validation
    #     if abs(threshold_price - fair_value) <= threshold_to_not_add:
    #         stored_data["fair_history"]["KELP"].append(fair_value)
    #         if len(stored_data["fair_history"]["KELP"]) > window_size:
    #             stored_data["fair_history"]["KELP"].pop(0)

    #     # Dynamic take width based on volatility
    #     if len(fair_history) > 1:
    #         volatility = np.std(fair_history[-window_size:])
    #         take_width = max(1, min(3, round(volatility * 0.5)))
    #     else:
    #         take_width = 1

    #     buy_order_volume = 0
    #     sell_order_volume = 0

    #     # Improved market taking with size-based scaling
    #     if order_depth.sell_orders:
    #         best_ask = min(order_depth.sell_orders.keys())
    #         best_ask_amount = -order_depth.sell_orders[best_ask]
    #         price_improvement = threshold_price - best_ask
            
    #         if best_ask <= threshold_price - take_width:
    #             # Scale order size based on price improvement
    #             scale_factor = min(1.5, 1 + (price_improvement / threshold_price))
    #             max_quantity = min(best_ask_amount, limit - position)
    #             quantity = int(max_quantity * scale_factor)
                
    #             if quantity > 0:
    #                 orders.append(Order("KELP", best_ask, quantity))
    #                 buy_order_volume += quantity

    #     # Similar improvements for selling
    #     if order_depth.buy_orders:
    #         best_bid = max(order_depth.buy_orders.keys())
    #         best_bid_amount = order_depth.buy_orders[best_bid]
    #         price_improvement = best_bid - threshold_price
            
    #         if best_bid >= threshold_price + take_width:
    #             scale_factor = min(1.5, 1 + (price_improvement / threshold_price))
    #             max_quantity = min(best_bid_amount, limit + position)
    #             quantity = int(max_quantity * scale_factor)
                
    #             if quantity > 0:
    #                 orders.append(Order("KELP", best_bid, -quantity))
    #                 sell_order_volume += quantity

    #     # Improved market making with dynamic spreads
    #     position_adjustment = 5 * (position / limit)  # Position-based price adjustment
    #     bid = round(threshold_price - dynamic_spread - position_adjustment)
    #     ask = round(threshold_price + dynamic_spread - position_adjustment)

    #     # Smart order sizing
    #     remaining_buy = limit - (position + buy_order_volume)
    #     remaining_sell = limit + (position - sell_order_volume)

    #     if remaining_buy > 0:
    #         orders.append(Order("KELP", bid, remaining_buy))
    #     if remaining_sell > 0:
    #         orders.append(Order("KELP", ask, -remaining_sell))

    #     return orders
    # def kelp(self, state: TradingState, limit: int, stored_data):
    #     orders: List[Order] = []

    #     order_depth = state.order_depths["KELP"]

    #     mm_ask = max(order_depth.sell_orders.keys())
    #     mm_bid = min(order_depth.buy_orders.keys())

    #     fair_value = (mm_ask + mm_bid) / 2
    #     buy_order_volume = 0
    #     sell_order_volume = 0
    #     window_size = 5
    #     threshold_to_not_add = 5

    #     fair_history = stored_data["fair_history"]["KELP"]
    #     if len(fair_history) == 0:
    #         threshold_price = fair_value
    #         stored_data["fair_history"]["KELP"].append(fair_value)
    #     elif len(fair_history) < window_size:
    #         threshold_price = sum(fair_history) / len(fair_history)
    #         stored_data["fair_history"]["KELP"].append(fair_value)
    #     else:
    #         threshold_price = sum(fair_history[-window_size:]) / window_size
    #         if abs(threshold_price - fair_value) <= threshold_to_not_add:
    #             stored_data["fair_history"]["KELP"].append(fair_value)

    #     position = state.position["KELP"] if "KELP" in state.position else 0

    #     take_width = 1

    #     if len(order_depth.sell_orders) != 0:
    #         # market taking
    #         best_ask = min(order_depth.sell_orders.keys())
    #         best_ask_amount = -1 * order_depth.sell_orders[best_ask]
    #         if best_ask <= threshold_price - take_width: #add conditions such that it scales to buy more when the buy seems more profitable
    #             quantity = min(best_ask_amount, limit - position)
    #             if quantity > 0:
    #                 orders.append(Order("KELP", best_ask, quantity))
    #                 buy_order_volume += quantity
    #                 order_depth.sell_orders[best_ask] += quantity
    #                 if order_depth.sell_orders[best_ask] == 0:
    #                     del order_depth.sell_orders[best_ask]

    #     if len(order_depth.buy_orders) != 0:
    #         best_bid = max(order_depth.buy_orders.keys())
    #         best_bid_amount = order_depth.buy_orders[best_bid]
    #         if best_bid >= threshold_price - take_width: #add conditions such that it scales to sell more when the sell seems more profitable
    #             quantity = min(best_bid_amount, limit + position)
    #             if quantity > 0:
    #                 orders.append(Order("KELP", best_bid, -1 * quantity))
    #                 sell_order_volume += quantity
    #                 order_depth.buy_orders[best_bid] -= quantity
    #                 if order_depth.buy_orders[best_bid] == 0:
    #                     del order_depth.buy_orders[best_bid]

    #     # make 0 ev trades to try to get back to 0 position
    #     orders, buy_order_volume, sell_order_volume = self.clear_orders(
    #         state,
    #         "KELP",
    #         buy_order_volume,
    #         sell_order_volume,
        #     threshold_price,
        #     take_width,
        #     limit,
        #     orders
        # )

        # # market making
        # # if prices are at most this much above/below the fair, do not market make
        # disregard_edge = 0

        # # if prices are at most this much above/below the fair, join (market make at the same price)
        # join_edge = 1
        # asks_above_fair = [price for price in order_depth.sell_orders.keys() if price > threshold_price + disregard_edge]
        # bids_below_fair = [price for price in order_depth.buy_orders.keys() if price < threshold_price - disregard_edge]

        # best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        # best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        # ask = threshold_price + 1
        # if best_ask_above_fair != None:
        #     # joining criteria
        #     if best_ask_above_fair - fair_value <= join_edge:
        #         ask = best_ask_above_fair
        #     # pennying criteria (undercutting by the minimum)
        #     else:
        #         ask = best_ask_above_fair - 1

        # bid = fair_value - 1
        # if best_bid_below_fair != None:
        #     if abs(fair_value - best_bid_below_fair) <= join_edge:
        #         bid = best_bid_below_fair
        #     else:
        #         bid = best_bid_below_fair + 1

        # # how many buy orders we could put out
        # buy_quantity = limit - (position + buy_order_volume)
        # if buy_quantity > 0:
        #     orders.append(Order("KELP", round(bid), buy_quantity))

        # sell_quantity = limit + (position - sell_order_volume)
        # if sell_quantity > 0:
        #     orders.append(Order("KELP", round(ask), -1 * sell_quantity))

        # return orders
    def forecast_fair_value(self, history: list[float], current_price: float) -> float:
        if len(history) < 6:
            return current_price

        ar_window = 5
        ma_window = 3

        ar_component = sum(history[-ar_window:]) / ar_window
        diff = history[-1] - history[-2]
        ma_component = sum(history[-i] - history[-i-1] for i in range(1, ma_window+1)) / ma_window

        predicted = 0.6 * ar_component + 0.2 * (history[-1] + diff) + 0.2 * (history[-1] + ma_component)
        fair_value = 0.7 * predicted + 0.3 * current_price

        return fair_value

    def momentum(self,
                 state: TradingState,
                 limit: int,
                 window: int,
                 momentum: float,
                 sticky,
                 stored_data,
                 orders: List[Order],
                 fair_value: float,
                 momentum_override: bool,
                 direction,
                 ):
        if len(stored_data["SQUID_INK"]["mins"]) < window:
            return False, None, orders

        order_depth = state.order_depths["SQUID_INK"]
        position = state.position["SQUID_INK"]

        stored_data["SQUID_INK"]["maxs"] = stored_data["SQUID_INK"]["maxs"][-(sticky * window):]
        stored_data["SQUID_INK"]["mins"] = stored_data["SQUID_INK"]["mins"][-(sticky * window):]
        stored_data["SQUID_INK"]["mm_prices"] = stored_data["SQUID_INK"]["mm_prices"][-window:]

        biggest_max = max(stored_data["SQUID_INK"]["maxs"][-window:-1])
        smallest_min = min(stored_data["SQUID_INK"]["mins"][-window:-1])
        sticky_max = max(stored_data["SQUID_INK"]["maxs"])
        sticky_min = min(stored_data["SQUID_INK"]["mins"])

        # if we are not momentum trading, check the following
        if not momentum_override:
            test_up = fair_value - (smallest_min + momentum)
            test_down = (biggest_max - momentum) - fair_value

            # if our fair is much higher than the smallest min, we are on an upswing and need to buy
            if fair_value >= smallest_min + momentum and test_up > test_down:
                best_ask = min(order_depth.sell_orders.keys())
                best_ask_amount = -1 * order_depth.sell_orders[best_ask]
                quantity = min(best_ask_amount, limit - position)
                orders.append(Order("SQUID_INK", best_ask, quantity))
                momentum_override = True
                direction = True

            elif fair_value <= biggest_max - momentum and test_down > test_up:
                best_bid = max(order_depth.buy_orders.keys())
                best_bid_amount = order_depth.buy_orders[best_bid]
                quantity = min(best_bid_amount, limit + position)
                orders.append(Order("SQUID_INK", best_bid, -1 * quantity))
                momentum_override = True
                direction = False

        else:
            if direction:
                # if we stopped swinging up, immediately stop momentum trading and sell
                if fair_value < sticky_min:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_amount = order_depth.buy_orders[best_bid]
                    quantity = min(best_bid_amount, limit + position)
                    orders.append(Order("SQUID_INK", best_bid, -1 * quantity))
                    momentum_override = False
                    direction = None

                # otherwise keep on buying
                else:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_amount = -1 * order_depth.sell_orders[best_ask]
                    quantity = min(best_ask_amount, limit - position)
                    orders.append(Order("SQUID_INK", best_ask, quantity))

            else:
                if fair_value > sticky_max:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_amount = -1 * order_depth.sell_orders[best_ask]
                    quantity = min(best_ask_amount, limit - position)
                    orders.append(Order("SQUID_INK", best_ask, quantity))
                    momentum_override = False
                    direction = None

                else:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_amount = order_depth.buy_orders[best_bid]
                    quantity = min(best_bid_amount, limit + position)
                    orders.append(Order("SQUID_INK", best_bid, -1 * quantity))

        return momentum_override, direction, orders
    # def kelp(self, state: TradingState, limit: int, stored_data):
    #     orders: List[Order] = []
    #     product = "KELP"

    #     if product not in state.order_depths:
    #         return []

    #     order_depth = state.order_depths[product]
    #     if not order_depth.sell_orders or not order_depth.buy_orders:
    #         return []

    #     best_bid = max(order_depth.buy_orders)
    #     best_ask = min(order_depth.sell_orders)
    #     current_price = (best_bid + best_ask) / 2

    #     # Store mid price
    #     stored_data[product].setdefault("price_history", []).append(current_price)
    #     if len(stored_data[product]["price_history"]) > 100:
    #         stored_data[product]["price_history"].pop(0)

    #     # Forecast fair value
    #     fair_value = self.forecast_fair_value(stored_data[product]["price_history"], current_price)

    #     position = state.position.get(product, 0)
    #     take_width = 1

    #     # Market taking
    #     if best_ask <= fair_value - take_width:
    #         quantity = min(-order_depth.sell_orders[best_ask], limit - position)
    #         if quantity > 0:
    #             orders.append(Order(product, best_ask, quantity))

    #     if best_bid >= fair_value + take_width:
    #         quantity = min(order_depth.buy_orders[best_bid], limit + position)
    #         if quantity > 0:
    #             orders.append(Order(product, best_bid, -quantity))

    #     # Market making
    #     spread = max(1, take_width * 2)
    #     bid_price = round(fair_value - spread)
    #     ask_price = round(fair_value + spread)

    #     buy_qty = limit - position
    #     sell_qty = limit + position

    #     if buy_qty > 0:
    #         orders.append(Order(product, bid_price, buy_qty))
    #     if sell_qty > 0:
    #         orders.append(Order(product, ask_price, -sell_qty))

    #     return orders
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
        disregard_edge = 1

        # if prices are at most this much above/below the fair, join (market make at the same price)
        join_edge = 2
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
    # def ink(self, state: TradingState, limit: int, stored_data):
    #     orders: List[Order] = []

    #     order_depth = state.order_depths["SQUID_INK"]
    #     position = state.position["SQUID_INK"] if "SQUID_INK" in state.position else 0

    #     buy_order_volume = 0
    #     sell_order_volume = 0

    #     # fair value calculation: it seems that the fair is the mid-price of the highest ask and lowest bid
    #     mm_ask = max(order_depth.sell_orders.keys())
    #     mm_bid = min(order_depth.buy_orders.keys())

    #     fair_value = (mm_ask + mm_bid) / 2
    #     take_width = 1

    #     # momentum trading calculation
    #     DONCHIAN_WINDOW = 5
    #     MOMENTUM = 5
    #     STICKY = 10
    #     MOMENTUM_OVERRIDE, DIRECTION = stored_data["SQUID_INK"]["momentum_override"]

    #     stored_data["SQUID_INK"]["maxs"].append(min(order_depth.sell_orders.keys()))
    #     stored_data["SQUID_INK"]["mins"].append(max(order_depth.buy_orders.keys()))
    #     stored_data["SQUID_INK"]["mm_prices"].append(fair_value)

    #     MOMENTUM_OVERRIDE, DIRECTION, orders = self.momentum(state,
    #                                               limit,
    #                                               DONCHIAN_WINDOW,
    #                                               MOMENTUM,
    #                                               STICKY,
    #                                               stored_data,
    #                                               orders,
    #                                               fair_value,
    #                                               MOMENTUM_OVERRIDE,
    #                                               DIRECTION
    #                                               )

    #     stored_data["SQUID_INK"]["momentum_override"] = (MOMENTUM_OVERRIDE, DIRECTION)

    #     # if not momentum trading, just kelp strat
    #     if not MOMENTUM_OVERRIDE:
    #         if len(order_depth.sell_orders) != 0:
    #             # market taking
    #             best_ask = min(order_depth.sell_orders.keys())
    #             best_ask_amount = -1 * order_depth.sell_orders[best_ask]

    #             if best_ask <= fair_value - take_width:
    #                 quantity = min(best_ask_amount, limit - position)
    #                 if quantity > 0:
    #                     orders.append(Order("SQUID_INK", best_ask, quantity))
    #                     buy_order_volume += quantity
    #                     order_depth.sell_orders[best_ask] += quantity
    #                     if order_depth.sell_orders[best_ask] == 0:
    #                         del order_depth.sell_orders[best_ask]

    #         if len(order_depth.buy_orders) != 0:
    #             best_bid = max(order_depth.buy_orders.keys())
    #             best_bid_amount = order_depth.buy_orders[best_bid]

    #             if best_bid >= fair_value + take_width:
    #                 quantity = min(best_bid_amount, limit + position)
    #                 if quantity > 0:
    #                     orders.append(Order("SQUID_INK", best_bid, -1 * quantity))
    #                     sell_order_volume += quantity
    #                     order_depth.buy_orders[best_bid] -= quantity
    #                     if order_depth.buy_orders[best_bid] == 0:
    #                         del order_depth.buy_orders[best_bid]

    #         # make 0 ev trades to try to get back to 0 position
    #         orders, buy_order_volume, sell_order_volume = self.clear_orders(
    #             state,
    #             "SQUID_INK",
    #             buy_order_volume,
    #             sell_order_volume,
    #             fair_value,
    #             take_width,
    #             limit,
    #             orders
    #         )

    #         # market making
    #         # if prices are at most this much above/below the fair, do not market make
    #         disregard_edge = 0

    #         # if prices are at most this much above/below the fair, join (market make at the same price)
    #         join_edge = 1
    #         asks_above_fair = [price for price in order_depth.sell_orders.keys() if price > fair_value + disregard_edge]
    #         bids_below_fair = [price for price in order_depth.buy_orders.keys() if price < fair_value - disregard_edge]

    #         best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
    #         best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

    #         ask = fair_value + 1
    #         if best_ask_above_fair != None:
    #             # joining criteria
    #             if best_ask_above_fair - fair_value <= join_edge:
    #                 ask = best_ask_above_fair
    #             # pennying criteria (undercutting by the minimum)
    #             else:
    #                 ask = best_ask_above_fair - 1

    #         bid = fair_value - 1
    #         if best_bid_below_fair != None:
    #             if abs(fair_value - best_bid_below_fair) <= join_edge:
    #                 bid = best_bid_below_fair
    #             else:
    #                 bid = best_bid_below_fair + 1

    #         # how many buy orders we could put out
    #         buy_quantity = limit - (position + buy_order_volume)
    #         if buy_quantity > 0:
    #             orders.append(Order("SQUID_INK", round(bid), buy_quantity))

    #         sell_quantity = limit + (position - sell_order_volume)
    #         if sell_quantity > 0:
    #             orders.append(Order("SQUID_INK", round(ask), -1 * sell_quantity))

    #         return orders

    #     else:
    #         return orders
    def ink(self, state: TradingState, limit: int, stored_data):
        orders: List[Order] = []
        product = "SQUID_INK"
        order_depth = state.order_depths[product]
        position = state.position.get(product, 0)

        if not order_depth.sell_orders or not order_depth.buy_orders:
            return []

        best_ask = min(order_depth.sell_orders)
        best_bid = max(order_depth.buy_orders)
        current_mid = (best_ask + best_bid) / 2

        # === Historical Price Update ===
        if "price_history" not in stored_data[product]:
            stored_data[product]["price_history"] = []
        price_history = stored_data[product]["price_history"]
        price_history.append(current_mid)
        if len(price_history) > 100:
            price_history.pop(0)

        # === Predictive Fair Value ===
        predicted_price = current_mid
        if len(price_history) >= 20:
            ar_window = 5
            ar_component = sum(price_history[-ar_window:]) / ar_window
            diff = price_history[-1] - price_history[-2] if len(price_history) > 1 else 0
            ma_window = 3
            ma_component = sum(price_history[-i] - price_history[-i-1] for i in range(1, ma_window+1)) / ma_window if len(price_history) >= ma_window + 1 else 0
            predicted_price = 0.6 * ar_component + 0.2 * (price_history[-1] + diff) + 0.2 * (price_history[-1] + ma_component)

        fair_value = 0.7 * predicted_price + 0.3 * current_mid

        # === Volatility-Adaptive Spread ===
        price_std = np.std(price_history[-10:]) if len(price_history) >= 10 else 0
        take_width = max(1, min(3, round(price_std * 0.67)))
        spread = max(1, take_width * 2)

        # === Position-Aware Skew ===
        skew = -min(2, position / 10) if position > 0 else min(2, -position / 10)

        # === Market Taking ===
        buy_order_volume = 0
        sell_order_volume = 0

        if best_ask <= fair_value - take_width:
            quantity = min(-order_depth.sell_orders[best_ask], limit - position)
            if quantity > 0:
                orders.append(Order(product, best_ask, quantity))
                buy_order_volume += quantity

        if best_bid >= fair_value + take_width:
            quantity = min(order_depth.buy_orders[best_bid], limit + position)
            if quantity > 0:
                orders.append(Order(product, best_bid, -quantity))
                sell_order_volume += quantity

        # === Market Making with Predicted Price Anchoring ===
        disregard_edge = 1
        join_edge = 2

        asks_above_pred = [p for p in order_depth.sell_orders if p > predicted_price + disregard_edge]
        bids_below_pred = [p for p in order_depth.buy_orders if p < predicted_price - disregard_edge]

        best_ask_above = min(asks_above_pred) if asks_above_pred else None
        best_bid_below = max(bids_below_pred) if bids_below_pred else None

        ask = round(predicted_price + spread + skew)
        if best_ask_above is not None:
            ask = best_ask_above if best_ask_above - predicted_price <= join_edge else best_ask_above - 1

        bid = round(predicted_price - spread + skew)
        if best_bid_below is not None:
            bid = best_bid_below if abs(predicted_price - best_bid_below) <= join_edge else best_bid_below + 1

        remaining_buy = limit - (position + buy_order_volume)
        remaining_sell = limit + (position - sell_order_volume)

        if remaining_buy > 0:
            orders.append(Order(product, bid, remaining_buy))
        if remaining_sell > 0:
            orders.append(Order(product, ask, -remaining_sell))

        return orders
    # def ink(self, state: TradingState, limit: int, stored_data):
    #     orders: List[Order] = []

    #     order_depth = state.order_depths["SQUID_INK"]
    #     position = state.position["SQUID_INK"] if "SQUID_INK" in state.position else 0

    #     # Calculate current fair value
    #     mm_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
    #     mm_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        
    #     if mm_ask is None or mm_bid is None:
    #         return orders  # No orders if market is empty
            
    #     current_mid_price = (mm_ask + mm_bid) / 2
        
    #     # Store historical prices (keep only the last 100 to manage memory)
    #     if "price_history" not in stored_data["SQUID_INK"]:
    #         stored_data["SQUID_INK"]["price_history"] = []
    #     stored_data["SQUID_INK"]["price_history"].append(current_mid_price)
    #     if len(stored_data["SQUID_INK"]["price_history"]) > 100:
    #         stored_data["SQUID_INK"]["price_history"].pop(0)
        
    #     # Simple ARIMA-like prediction (we'll use a simplified version since we can't import statsmodels)
    #     predicted_price = current_mid_price  # Default to current price if we can't predict
        
    #     if len(stored_data["SQUID_INK"]["price_history"]) >= 20:  # Need enough data points
    #         history = stored_data["SQUID_INK"]["price_history"]
            
    #         # Simple moving average (AR component)
    #         ar_window = 5
    #         ar_component = sum(history[-ar_window:]) / ar_window
            
    #         # Simple difference (I component)
    #         diff = history[-1] - history[-2] if len(history) > 1 else 0
            
    #         # Simple momentum (MA component)
    #         ma_window = 3
    #         if len(history) >= ma_window + 1:
    #             ma_component = sum(history[-i] - history[-i-1] for i in range(1, ma_window+1)) / ma_window
    #         else:
    #             ma_component = 0
            
    #         # Combine components with weights (these weights can be adjusted)
    #         predicted_price = 0.6 * ar_component + 0.2 * (history[-1] + diff) + 0.2 * (history[-1] + ma_component)
        
    #     # Calculate fair value as weighted average between current mid and predicted price
    #     fair_value = 0.7 * predicted_price + 0.3 * current_mid_price
        
    #     # Dynamic take width based on recent volatility
    #     if len(stored_data["SQUID_INK"]["price_history"]) >= 10:
    #         recent_prices = stored_data["SQUID_INK"]["price_history"][-10:]
    #         price_std = np.std(recent_prices)
    #         take_width = max(1, min(3, round(price_std * 0.67)))  # Clamped between 1 and 3
    #     else:
    #         take_width = 1
        
    #     buy_order_volume = 0
    #     sell_order_volume = 0

    #     # Market taking - buy if ask is below fair value minus take width
    #     if order_depth.sell_orders:
    #         best_ask = min(order_depth.sell_orders.keys())
    #         best_ask_amount = -order_depth.sell_orders[best_ask]
            
    #         if best_ask <= fair_value - take_width:
    #             quantity = min(best_ask_amount, limit - position)
    #             if quantity > 0:
    #                 orders.append(Order("SQUID_INK", best_ask, quantity))
    #                 buy_order_volume += quantity

    #     # Market taking - sell if bid is above fair value plus take width
    #     if order_depth.buy_orders:
    #         best_bid = max(order_depth.buy_orders.keys())
    #         best_bid_amount = order_depth.buy_orders[best_bid]
            
    #         if best_bid >= fair_value + take_width:
    #             quantity = min(best_bid_amount, limit + position)
    #             if quantity > 0:
    #                 orders.append(Order("SQUID_INK", best_bid, -quantity))
    #                 sell_order_volume += quantity

    #     # Position clearing logic
    #     orders, buy_order_volume, sell_order_volume = self.clear_orders(
    #         state,
    #         "SQUID_INK",
    #         buy_order_volume,
    #         sell_order_volume,
    #         fair_value,
    #         take_width,
    #         limit,
    #         orders
    #     )
    #     disregard_edge = 1

    #     # if prices are at most this much above/below the fair, join (market make at the same price)
    #     join_edge = 2
    #     asks_above_fair = [price for price in order_depth.sell_orders.keys() if price > fair_value + disregard_edge]
    #     bids_below_fair = [price for price in order_depth.buy_orders.keys() if price < fair_value - disregard_edge]

    #     best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
    #     best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

    #     ask = fair_value + 1
    #     if best_ask_above_fair != None:
    #         # joining criteria
    #         if best_ask_above_fair - fair_value <= join_edge:
    #             ask = best_ask_above_fair
    #         # pennying criteria (undercutting by the minimum)
    #         else:
    #             ask = best_ask_above_fair - 1

    #     bid = fair_value - 1
    #     if best_bid_below_fair != None:
    #         if abs(fair_value - best_bid_below_fair) <= join_edge:
    #             bid = best_bid_below_fair
    #         else:
    #             bid = best_bid_below_fair + 1

    #     # how many buy orders we could put out
    #     buy_quantity = limit - (position + buy_order_volume)
    #     if buy_quantity > 0:
    #         orders.append(Order("SQUID_INK", round(bid), buy_quantity))

    #     sell_quantity = limit + (position - sell_order_volume)
    #     if sell_quantity > 0:
    #         orders.append(Order("SQUID_INK", round(ask), -1 * sell_quantity))
    #     return orders


    # def ink(self, state: TradingState, limit: int, stored_data):
    #     orders: List[Order] = []

    #     order_depth = state.order_depths["SQUID_INK"]
    #     position = state.position["SQUID_INK"] if "SQUID_INK" in state.position else 0

    #     # Calculate current fair value
    #     mm_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
    #     mm_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        
    #     if mm_ask is None or mm_bid is None:
    #         return orders  # No orders if market is empty
            
    #     current_mid_price = (mm_ask + mm_bid) / 2
        
    #     # Store historical prices (keep only the last 100 to manage memory)
    #     if "price_history" not in stored_data["SQUID_INK"]:
    #         stored_data["SQUID_INK"]["price_history"] = []
    #     stored_data["SQUID_INK"]["price_history"].append(current_mid_price)
    #     if len(stored_data["SQUID_INK"]["price_history"]) > 100:
    #         stored_data["SQUID_INK"]["price_history"].pop(0)
        
    #     # Simple ARIMA-like prediction (we'll use a simplified version since we can't import statsmodels)
    #     predicted_price = current_mid_price  # Default to current price if we can't predict
        
    #     if len(stored_data["SQUID_INK"]["price_history"]) >= 20:  # Need enough data points
    #         history = stored_data["SQUID_INK"]["price_history"]
            
    #         # Simple moving average (AR component)
    #         ar_window = 5
    #         ar_component = sum(history[-ar_window:]) / ar_window
            
    #         # Simple difference (I component)
    #         diff = history[-1] - history[-2] if len(history) > 1 else 0
            
    #         # Simple momentum (MA component)
    #         ma_window = 3
    #         if len(history) >= ma_window + 1:
    #             ma_component = sum(history[-i] - history[-i-1] for i in range(1, ma_window+1)) / ma_window
    #         else:
    #             ma_component = 0
            
    #         # Combine components with weights (these weights can be adjusted)
    #         predicted_price = 0.6 * ar_component + 0.2 * (history[-1] + diff) + 0.2 * (history[-1] + ma_component)
        
    #     # Calculate fair value as weighted average between current mid and predicted price
    #     fair_value = 0.7 * predicted_price + 0.3 * current_mid_price
        
    #     # Dynamic take width based on recent volatility
    #     if len(stored_data["SQUID_INK"]["price_history"]) >= 10:
    #         recent_prices = stored_data["SQUID_INK"]["price_history"][-10:]
    #         price_std = np.std(recent_prices)
    #         take_width = max(1, min(3, round(price_std * 0.67)))  # Clamped between 1 and 3
    #     else:
    #         take_width = 1
        
    #     buy_order_volume = 0
    #     sell_order_volume = 0

    #     # Market taking - buy if ask is below fair value minus take width
    #     if order_depth.sell_orders:
    #         best_ask = min(order_depth.sell_orders.keys())
    #         best_ask_amount = -order_depth.sell_orders[best_ask]
            
    #         if best_ask <= fair_value - take_width:
    #             quantity = min(best_ask_amount, limit - position)
    #             if quantity > 0:
    #                 orders.append(Order("SQUID_INK", best_ask, quantity))
    #                 buy_order_volume += quantity

    #     # Market taking - sell if bid is above fair value plus take width
    #     if order_depth.buy_orders:
    #         best_bid = max(order_depth.buy_orders.keys())
    #         best_bid_amount = order_depth.buy_orders[best_bid]
            
    #         if best_bid >= fair_value + take_width:
    #             quantity = min(best_bid_amount, limit + position)
    #             if quantity > 0:
    #                 orders.append(Order("SQUID_INK", best_bid, -quantity))
    #                 sell_order_volume += quantity

    #     # Position clearing logic
    #     orders, buy_order_volume, sell_order_volume = self.clear_orders(
    #         state,
    #         "SQUID_INK",
    #         buy_order_volume,
    #         sell_order_volume,
    #         fair_value,
    #         take_width,
    #         limit,
    #         orders
    #     )
        #New Market making strategy
        # Market making with dynamic spreads based on volatility
        # spread = max(1, take_width * 2)  # Spread is twice the take width
        # position_adjustment = 0
        
        # # Adjust prices based on current position
        # if position > 0:
        #     position_adjustment = -min(2, position / 10)  # Lower prices if long
        # elif position < 0:
        #     position_adjustment = min(2, -position / 10)  # Raise prices if short

        # bid_price = round(fair_value - spread + position_adjustment)
        # ask_price = round(fair_value + spread + position_adjustment)

        # # Calculate remaining quantities we can trade
        # remaining_buy = limit - (position + buy_order_volume)
        # remaining_sell = limit + (position - sell_order_volume)

        # # Post market making orders
        # if remaining_buy > 0:
        #     orders.append(Order("SQUID_INK", bid_price, remaining_buy))
        
        # if remaining_sell > 0:
        #     orders.append(Order("SQUID_INK", ask_price, -remaining_sell))

        # return orders

    def run(self, state: TradingState):
        stored_data = json.loads(state.traderData) if state.traderData else {}

        for product in ["RAINFOREST_RESIN", "KELP", "SQUID_INK"]:
            if product not in stored_data:
                stored_data[product] = {}
            if "mins" not in stored_data[product]:
                stored_data[product]["mins"] = []
            if "maxs" not in stored_data[product]:
                stored_data[product]["maxs"] = []
            if "mm_prices" not in stored_data[product]:
                stored_data[product]["mm_prices"] = []
            # the pair[0] means True if we are in momentum, pair[1] means True if we are swinging up, False if down
            if "momentum_override" not in stored_data[product]:
                stored_data[product]["momentum_override"] = (False, None)
            if "fair_history" not in stored_data:
                stored_data["fair_history"] = {
                    "SQUID_INK": [],
                    "KELP": [],
                    "RAINFOREST_RESIN": []
                }

        POSITION_LIMITS = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
        }

        result = {}

        for product in ["RAINFOREST_RESIN", "KELP", "SQUID_INK"]:
            if product == "RAINFOREST_RESIN":
                result["RAINFOREST_RESIN"] = self.resin(state, POSITION_LIMITS["RAINFOREST_RESIN"])
            elif product == "KELP":
                result["KELP"] = self.kelp(state, POSITION_LIMITS["KELP"])
            elif product == "SQUID_INK":
                result["SQUID_INK"] = self.ink(state, POSITION_LIMITS["SQUID_INK"], stored_data)

        trader_data = json.dumps(stored_data, separators = (",", ":"))

        conversions = 0
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data