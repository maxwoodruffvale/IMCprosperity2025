from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List
import string
import jsonpickle
import numpy as np
import math

class Trader:
    def __init__(self):
        self.kelp_prices = []
        self.kelp_vwap = []
        # vwap = Volume Weighted Average Price

    # if you reach position limit, you can make 0 ev trades (based on your fair value) to open up more position
    # this adds those 0 ev orders to the final result btw
    def clear_position_order(
            self,
            orders: List[Order],
            order_depth: OrderDepth,
            position: int,
            position_limit: int,
            product: str,
            buy_order_volume: int,
            sell_order_volume: int,
            fair_value: float
    ):

        # remember market taking is the initial part of the timestamp: bots make trades, order book is set up
        # then you come in and ideally grab up all the +ev trades (market taking)
        # then you place +ev orders (market making)
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)

        # how much we have left after the takes
        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        # essentially if we are long after the takes, we want to sell at 0 ev (fair price) to get back to 0 position
        if position_after_take > 0:
            if fair_for_ask in order_depth.buy_orders.keys():

                # how much is possible to clear
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)

                # make sure that while selling we dont go over short limit
                sent_quantity = min(sell_quantity, clear_quantity)

                #
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            if fair_for_bid in order_depth.sell_orders.keys():
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
                sent_quantity = min(buy_quantity, clear_quantity)
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    # resin market taking/making
    def resin_orders(
            self,
            order_depth: OrderDepth,
            fair_value: int,
            position: int,
            position_limit: int
    ) -> List[Order]:
        orders: List[Order] = []

        buy_order_volume = 0
        sell_order_volume = 0

        # best ask above fair (i think)
        baaf = min([price for price in order_depth.sell_orders.keys() if price > fair_value + 1])
        # best bid below fair
        bbbf = max([price for price in order_depth.buy_orders.keys() if price < fair_value - 1])

        # we buy if best ask is below fair (market taking)
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if best_ask < fair_value:
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_ask, quantity))
                    buy_order_volume += quantity

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > fair_value:
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_bid, -1 * quantity))
                    sell_order_volume += quantity

        # put in the 0 ev trades to get as close to 0 position as possible
        buy_order_volume, sell_order_volume = self.clear_position_order(
            orders,
            order_depth,
            position,
            position_limit,
            "RAINFOREST_RESIN",
            buy_order_volume,
            sell_order_volume,
            fair_value,
        )

        # market making after we snatch up the good stuff
        # how much room we have left after all the buying
        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", bbbf + 1, buy_quantity))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", baaf - 1, -sell_quantity))

        return orders

    # kelp helper function
    def kelp_fair_value(self):
        return None

    # kelp market taking/making
    def kelp_orders(
            self,
            order_depth: OrderDepth
    ) -> List[Order]:
        orders: List[Order] = []

        return orders

    def run(self, state: TradingState):
        result = {}

        resin_fair_value = 10000
        resin_position_limit = 50

        kelp_make_width = 3.5
        kelp_take_width = 1
        kelp_timespan = 10
        kelp_position_limit = 50

        if "RAINFOREST_RESIN" in state.order_depths:
            resin_position = state.position["RAINFOREST_RESIN"] if "RAINFOREST_RESIN" in state.position else 0
            resin_orders = self.resin_orders(
                state.order_depths["RAINFOREST_RESIN"], resin_fair_value, resin_position, resin_position_limit
            )
            result["RAINFOREST_RESIN"] = resin_orders

        if "KELP" in state.order_depths:
            kelp_position = state.position["KELP"] if "KELP" in state.position else 0
            kelp_orders = self.kelp_orders(
                state.order_depths["KELP"]
            )

        trader_data = jsonpickle.encode(
            {
                "kelp_prices": self.kelp_prices,
                "kelp_vwap": self.kelp_vwap
            }
        )

        conversions = 1
        return result, conversions, trader_data
