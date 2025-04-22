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

@dataclass
class ProductData:
    end_pos: int
    buy_sum: int
    sell_sum: int
    bid_prices: List[float]
    bid_volumes: List[int]
    ask_prices: List[float]
    ask_volumes: List[int]
    fair_price: Optional[float]

    @classmethod
    def from_state(cls, product: str, state) -> "ProductData":
        od = state.order_depths[product]
        bids, asks = od.buy_orders, od.sell_orders
        position = state.position.get(product, 0)
        end_pos = state.position.get(product, 0)
        buy_sum = 200 - position
        sell_sum = 200 + position
        bid_prices  = list(bids.keys())
        bid_volumes = list(bids.values())
        ask_prices  = list(asks.keys())
        ask_volumes = list(asks.values())

        mm_bid = max(bids, key=bids.get) if bids else None
        mm_ask = min(asks, key=asks.get) if asks else None

        if mm_bid is not None and mm_ask is not None:
            fair = (mm_bid + mm_ask) / 2
        elif mm_ask is not None:
            fair = mm_ask
        elif mm_bid is not None:
            fair = mm_bid
        else:
            fair = None

        return cls(
            end_pos=end_pos,
            buy_sum=buy_sum,
            sell_sum=sell_sum,
            bid_prices=bid_prices,
            bid_volumes=bid_volumes,
            ask_prices=ask_prices,
            ask_volumes=ask_volumes,
            fair_price=fair
        )


PARAMS = {
    "VOLCANIC_ROCK_VOUCHER_9500":  {"returns_history_window": 13, "K": 9500,  "base_coef": 0.147604, "linear_coef": 0.010031, "squared_coef": 0.264416, "threshold": 0.0005},
    "VOLCANIC_ROCK_VOUCHER_9750":  {"returns_history_window": 13, "K": 9750,  "base_coef": 0.147604, "linear_coef": 0.010031, "squared_coef": 0.264416, "threshold": 0.0055},
    "VOLCANIC_ROCK_VOUCHER_10000": {"returns_history_window": 20, "K": 10000, "base_coef": 0.14786181, "linear_coef": 0.00099561, "squared_coef": 0.23544086, "threshold": 0.0035},
    "VOLCANIC_ROCK_VOUCHER_10250": {"returns_history_window": 13, "K": 10250, "base_coef": 0.147604, "linear_coef": 0.010031, "squared_coef": 0.264416, "threshold": 0.00055},
    "VOLCANIC_ROCK_VOUCHER_10500": {"returns_history_window": 13, "K": 10500, "base_coef": 0.147604, "linear_coef": 0.010031, "squared_coef": 0.264416, "threshold": 0.0005},
}
CURRENT_DAY = 4

class Trader:
    def __init__(self):
        self.returns_history = {v: deque(maxlen=PARAMS[v]["returns_history_window"]) for v in PARAMS.keys()}
        self.ink_olivia = ""
        self.picnic_olivia = {"CROISSANTS": "", "PICNIC_BASKET1": "", "PICNIC_BASKET2": ""}
        self.resume_picnic_arb = True
        self.internal_position = {"CROISSANTS": 0, "JAMS": 0, "DJEMBES": 0, "PICNIC_BASKET1": 0, "PICNIC_BASKET2": 0}
        
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
            clear_quantity = sum(
                abs(volume) for price, volume in order_depth.sell_orders.items() if price <= fair_for_bid)
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, sent_quantity))
                buy_order_volume += abs(sent_quantity)

        return orders, buy_order_volume, sell_order_volume
        
    def ink_strategy(self, state: TradingState, limit: int) -> List[Order]:
        orders: List[Order] = []

        order_depth = state.order_depths["SQUID_INK"]
        position = state.position["SQUID_INK"] if "SQUID_INK" in state.position else 0
        trades = state.market_trades.get("SQUID_INK", [])
        trades = [t for t in trades if t.timestamp == state.timestamp-100]   
  
        if any(t.buyer == "Olivia" for t in trades):
            self.ink_olivia = "buy"
            
        if any(t.seller == "Olivia" for t in trades):
            self.ink_olivia = "sell"
            
        if self.ink_olivia == "buy":
            if position == limit:
                self.ink_olivia = ""
                return orders
            
            for ask_price in sorted(order_depth.sell_orders.keys()):
                best_ask_amount = -1 * order_depth.sell_orders[ask_price] 
                quantity = min(best_ask_amount, limit - position)
                orders.append(Order("SQUID_INK", ask_price, quantity))

                position += quantity
                if quantity == 0:
                    break

        elif self.ink_olivia == "sell":
            if position == -limit:
                self.ink_olivia = ""
                return orders
            
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                best_bid_amount = order_depth.buy_orders[bid_price]
                quantity = min(best_bid_amount, limit + position)
                orders.append(Order("SQUID_INK", bid_price, -1 * quantity))

                position -= quantity
                if quantity == 0:
                    break
            
        return orders

    def convert_synthetic1_orders(self, synthetic_order_depth, state: TradingState, synthetic_order):
        best_bid = max(synthetic_order_depth.buy_orders.keys())
        best_ask = min(synthetic_order_depth.sell_orders.keys())

        price = synthetic_order.price
        quantity = synthetic_order.quantity

        order_depths = state.order_depths

        # buying synthetic
        if quantity > 0:
            croissants_price = min(order_depths["CROISSANTS"].sell_orders.keys())
            jams_price = min(order_depths["JAMS"].sell_orders.keys())
            djembes_price = min(order_depths["DJEMBES"].sell_orders.keys())

        # selling synthetic
        elif quantity < 0:
            croissants_price = max(order_depths["CROISSANTS"].buy_orders.keys())
            jams_price = max(order_depths["JAMS"].buy_orders.keys())
            djembes_price = max(order_depths["DJEMBES"].buy_orders.keys())

        croissants_order = Order("CROISSANTS", croissants_price, quantity * 6)
        self.internal_position["CROISSANTS"] += quantity * 6
        jams_order = Order("JAMS", jams_price, quantity * 3)
        self.internal_position["JAMS"] += quantity * 3
        djembes_order = Order("DJEMBES", djembes_price, quantity)
        self.internal_position["DJEMBES"] += quantity

        return croissants_order, jams_order, djembes_order

    # get the order depth of the synthetic1
    def synthetic1_order_depth(self, state: TradingState):
        order_depths = state.order_depths
        synthetic_order_depth = OrderDepth()

        croissants_best_ask = min(order_depths["CROISSANTS"].sell_orders.keys())
        croissants_best_bid = max(order_depths["CROISSANTS"].buy_orders.keys())

        jams_best_ask = min(order_depths["JAMS"].sell_orders.keys())
        jams_best_bid = max(order_depths["JAMS"].buy_orders.keys())

        djembes_best_ask = min(order_depths["DJEMBES"].sell_orders.keys())
        djembes_best_bid = max(order_depths["DJEMBES"].buy_orders.keys())

        synthetic_best_ask = croissants_best_ask * 6 + jams_best_ask * 3 + djembes_best_ask
        synthetic_best_bid = croissants_best_bid * 6 + jams_best_bid * 3 + djembes_best_bid

        croissants_ask_volume = -1 * order_depths["CROISSANTS"].sell_orders[croissants_best_ask]
        croissants_bid_volume = order_depths["CROISSANTS"].buy_orders[croissants_best_bid]
        try:
            croissants_position = self.internal_position["CROISSANTS"]
        except:
            croissants_position = 0

        jams_ask_volume = -1 * order_depths["JAMS"].sell_orders[jams_best_ask]
        jams_bid_volume = order_depths["JAMS"].buy_orders[jams_best_bid]
        try:
            jams_position = self.internal_position["JAMS"]
        except:
            jams_position = 0

        djembes_ask_volume = -1 * order_depths["DJEMBES"].sell_orders[djembes_best_ask]
        djembes_bid_volume = order_depths["DJEMBES"].buy_orders[djembes_best_bid]
        try:
            djembes_position = self.internal_position["DJEMBES"]
        except:
            djembes_position = 0

        synthetic_ask_volume = min(int(croissants_ask_volume / 6), int((250 - croissants_position) / 6),
                                   int(jams_ask_volume / 3), int((350 - jams_position) / 3),
                                   int(djembes_ask_volume), int(60 - djembes_position),
                                   )
        synthetic_bid_volume = min(int(croissants_bid_volume / 6), int((250 + croissants_position) / 6),
                                   int(jams_bid_volume / 3), int((350 + jams_position) / 3),
                                   int(djembes_bid_volume), int(60 + djembes_position),
                                   )

        synthetic_order_depth.sell_orders[synthetic_best_ask] = -1 * synthetic_ask_volume
        synthetic_order_depth.buy_orders[synthetic_best_bid] = synthetic_bid_volume

        return synthetic_order_depth

    def picnic1_strategy(self, state: TradingState, limit: int, stored_data, first: bool):
        orders: List[Order] = []
        croissants_orders: List[Order] = []
        jams_orders: List[Order] = []
        djembes_orders: List[Order] = []

        order_depth = state.order_depths["PICNIC_BASKET1"]
        position = self.internal_position["PICNIC_BASKET1"] if "PICNIC_BASKET1" in self.internal_position else 0
        # position = state.position["PICNIC_BASKET1"] if "PICNIC_BASKET1" in state.position else 0

        # fair value calculation: it seems that the fair is the mid-price of the highest ask and lowest bid
        picnic_fair_value = stored_data["PICNIC_BASKET1"]["fair_value"]

        # "fair" value for synthetic
        synthetic_order_depth = self.synthetic1_order_depth(state)

        synthetic_ask = max(synthetic_order_depth.sell_orders.keys())
        synthetic_bid = min(synthetic_order_depth.buy_orders.keys())

        synthetic_fair_value = (synthetic_ask + synthetic_bid) / 2

        take_width = 1
        buy_order_volume = 0
        sell_order_volume = 0

        # z score for spread based on rolling window
        z_trade = True
        window = 25
        z_score_threshold = 20

        spread = picnic_fair_value - synthetic_fair_value

        if first:
            stored_data["PICNIC_BASKET1"]["spread_history"].append(spread)

        if len(stored_data["PICNIC_BASKET1"]["spread_history"]) < 5:
            z_trade = False
        elif len(stored_data["PICNIC_BASKET1"]["spread_history"]) > window:
            stored_data["PICNIC_BASKET1"]["spread_history"].pop(0)

        if z_trade:
            spread_std = np.std(stored_data["PICNIC_BASKET1"]["spread_history"])

            if spread_std == 0:
                z_score = float("inf")
            else:
                z_score = (spread + 105.748375) / spread_std

            # if we are much higher, then picnic is overvalued and we should sell picnic, buy synthetic
            if z_score > z_score_threshold:
                if position > -1 * limit:
                    target_quantity = position + limit

                    try:
                        best_bid = max(order_depth.buy_orders.keys())
                    except:
                        return orders, croissants_orders, jams_orders, djembes_orders

                    best_bid_volume = order_depth.buy_orders[best_bid]

                    synthetic_ask_volume = -1 * synthetic_order_depth.sell_orders[synthetic_ask]

                    orderbook_volume = min(best_bid_volume, synthetic_ask_volume)
                    execute_volume = min(orderbook_volume, target_quantity)

                    if execute_volume == 0:
                        return orders, croissants_orders, jams_orders, djembes_orders

                    orders.append(Order("PICNIC_BASKET1", best_bid, -1 * execute_volume))
                    self.internal_position["PICNIC_BASKET1"] -= execute_volume

                    croissants_order, jams_order, djembes_order = self.convert_synthetic1_orders(synthetic_order_depth,
                                                                                                 state,
                                                                                                 Order("SYNTHETIC1",
                                                                                                       synthetic_ask,
                                                                                                       execute_volume),
                                                                                                 )

                    croissants_orders.append(croissants_order)
                    jams_orders.append(jams_order)
                    djembes_orders.append(djembes_order)

                    order_depth.buy_orders[best_bid] -= execute_volume
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]

                    state.order_depths["CROISSANTS"].sell_orders[croissants_order.price] += (6 * execute_volume)
                    if state.order_depths["CROISSANTS"].sell_orders[croissants_order.price] == 0:
                        del state.order_depths["CROISSANTS"].sell_orders[croissants_order.price]

                    state.order_depths["JAMS"].sell_orders[jams_order.price] += (3 * execute_volume)
                    if state.order_depths["JAMS"].sell_orders[jams_order.price] == 0:
                        del state.order_depths["JAMS"].sell_orders[jams_order.price]

                    state.order_depths["DJEMBES"].sell_orders[djembes_order.price] += execute_volume
                    if state.order_depths["DJEMBES"].sell_orders[djembes_order.price] == 0:
                        del state.order_depths["DJEMBES"].sell_orders[djembes_order.price]

                    sell_order_volume += execute_volume

            # z score too low, picnic undervalued, buy picnic, sell synthetic
            elif z_score < -1 * z_score_threshold:
                if position < limit:
                    target_quantity = limit - position

                    try:
                        best_ask = min(order_depth.sell_orders.keys())
                    except:
                        return orders, croissants_orders, jams_orders, djembes_orders

                    best_ask_volume = -1 * order_depth.sell_orders[best_ask]

                    synthetic_bid_volume = synthetic_order_depth.buy_orders[synthetic_bid]

                    orderbook_volume = min(best_ask_volume, synthetic_bid_volume)
                    execute_volume = min(orderbook_volume, target_quantity)

                    if execute_volume == 0:
                        return orders, croissants_orders, jams_orders, djembes_orders

                    orders.append(Order("PICNIC_BASKET1", best_ask, execute_volume))
                    self.internal_position["PICNIC_BASKET1"] += execute_volume

                    croissants_order, jams_order, djembes_order = self.convert_synthetic1_orders(synthetic_order_depth,
                                                                                                 state,
                                                                                                 Order("SYNTHETIC1",
                                                                                                       synthetic_bid,
                                                                                                       -1 * execute_volume),
                                                                                                 )

                    croissants_orders.append(croissants_order)
                    jams_orders.append(jams_order)
                    djembes_orders.append(djembes_order)

                    order_depth.sell_orders[best_ask] += execute_volume
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]

                    state.order_depths["CROISSANTS"].buy_orders[croissants_order.price] -= (6 * execute_volume)
                    if state.order_depths["CROISSANTS"].buy_orders[croissants_order.price] == 0:
                        del state.order_depths["CROISSANTS"].buy_orders[croissants_order.price]

                    state.order_depths["JAMS"].buy_orders[jams_order.price] -= (3 * execute_volume)
                    if state.order_depths["JAMS"].buy_orders[jams_order.price] == 0:
                        del state.order_depths["JAMS"].buy_orders[jams_order.price]

                    state.order_depths["DJEMBES"].buy_orders[djembes_order.price] -= execute_volume
                    if state.order_depths["DJEMBES"].buy_orders[djembes_order.price] == 0:
                        del state.order_depths["DJEMBES"].buy_orders[djembes_order.price]

                    buy_order_volume += execute_volume

        return orders, croissants_orders, jams_orders, djembes_orders

    def convert_synthetic2_orders(self, synthetic_order_depth, state: TradingState, synthetic_order):
        best_bid = max(synthetic_order_depth.buy_orders.keys())
        best_ask = min(synthetic_order_depth.sell_orders.keys())

        price = synthetic_order.price
        quantity = synthetic_order.quantity

        order_depths = state.order_depths

        # buying synthetic
        if quantity > 0:
            croissants_price = min(order_depths["CROISSANTS"].sell_orders.keys())
            jams_price = min(order_depths["JAMS"].sell_orders.keys())

        # selling synthetic
        elif quantity < 0:
            croissants_price = max(order_depths["CROISSANTS"].buy_orders.keys())
            jams_price = max(order_depths["JAMS"].buy_orders.keys())

        croissants_order = Order("CROISSANTS", croissants_price, quantity * 4)
        self.internal_position["CROISSANTS"] += quantity * 4

        jams_order = Order("JAMS", jams_price, quantity * 2)
        self.internal_position["JAMS"] += quantity * 2

        return croissants_order, jams_order

    # get the order depth of the synthetic2
    def synthetic2_order_depth(self, state: TradingState):
        order_depths = state.order_depths
        synthetic_order_depth = OrderDepth()

        croissants_best_ask = min(order_depths["CROISSANTS"].sell_orders.keys())
        croissants_best_bid = max(order_depths["CROISSANTS"].buy_orders.keys())

        jams_best_ask = min(order_depths["JAMS"].sell_orders.keys())
        jams_best_bid = max(order_depths["JAMS"].buy_orders.keys())

        synthetic_best_ask = croissants_best_ask * 4 + jams_best_ask * 2
        synthetic_best_bid = croissants_best_bid * 4 + jams_best_bid * 2

        croissants_ask_volume = -1 * order_depths["CROISSANTS"].sell_orders[croissants_best_ask]
        croissants_bid_volume = order_depths["CROISSANTS"].buy_orders[croissants_best_bid]
        try:
            croissants_position = self.internal_position["CROISSANTS"]
        except:
            croissants_position = 0

        jams_ask_volume = -1 * order_depths["JAMS"].sell_orders[jams_best_ask]
        jams_bid_volume = order_depths["JAMS"].buy_orders[jams_best_bid]
        try:
            jams_position = self.internal_position["JAMS"]
        except:
            jams_position = 0

        synthetic_ask_volume = min(int(croissants_ask_volume / 4), int((250 - croissants_position) / 4),
                                   int(jams_ask_volume / 2), int((350 - jams_position) / 2),
                                   )
        synthetic_bid_volume = min(int(croissants_bid_volume / 4), int((250 + croissants_position) / 4),
                                   int(jams_bid_volume / 2), int((350 + jams_position) / 2),
                                   )

        synthetic_order_depth.sell_orders[synthetic_best_ask] = -1 * synthetic_ask_volume
        synthetic_order_depth.buy_orders[synthetic_best_bid] = synthetic_bid_volume

        return synthetic_order_depth

    def picnic2_strategy(self, state: TradingState, limit: int, stored_data, first: bool):
        orders: List[Order] = []
        croissants_orders: List[Order] = []
        jams_orders: List[Order] = []

        order_depth = state.order_depths["PICNIC_BASKET2"]
        position = self.internal_position["PICNIC_BASKET2"] if "PICNIC_BASKET2" in self.internal_position else 0
        # position = state.position["PICNIC_BASKET2"] if "PICNIC_BASKET2" in state.position else 0

        picnic_fair_value = stored_data["PICNIC_BASKET2"]["fair_value"]

        # "fair" value for synthetic
        synthetic_order_depth = self.synthetic2_order_depth(state)

        synthetic_ask = max(synthetic_order_depth.sell_orders.keys())
        synthetic_bid = min(synthetic_order_depth.buy_orders.keys())

        synthetic_fair_value = (synthetic_ask + synthetic_bid) / 2

        take_width = 1
        buy_order_volume = 0
        sell_order_volume = 0

        # z score for spread based on rolling window 10/30 220402
        z_trade = True
        window = 25
        z_score_threshold = 20

        spread = picnic_fair_value - synthetic_fair_value

        if first:
            stored_data["PICNIC_BASKET2"]["spread_history"].append(spread)

        if len(stored_data["PICNIC_BASKET2"]["spread_history"]) < 5:
            z_trade = False
        elif len(stored_data["PICNIC_BASKET2"]["spread_history"]) > window:
            stored_data["PICNIC_BASKET2"]["spread_history"].pop(0)

        if z_trade:
            spread_std = np.std(stored_data["PICNIC_BASKET2"]["spread_history"])

            if spread_std == 0:
                z_score = float("inf")
            else:
                z_score = (spread - 96.50535) / spread_std

            # if we are much higher, then picnic is overvalued and we should sell picnic, buy synthetic
            if z_score > z_score_threshold:
                if position > -1 * limit:
                    target_quantity = position + limit

                    try:
                        best_bid = max(order_depth.buy_orders.keys())
                    except:
                        return orders, croissants_orders, jams_orders

                    best_bid_volume = order_depth.buy_orders[best_bid]

                    synthetic_ask_volume = -1 * synthetic_order_depth.sell_orders[synthetic_ask]

                    orderbook_volume = min(best_bid_volume, synthetic_ask_volume)
                    execute_volume = min(orderbook_volume, target_quantity)

                    if execute_volume == 0:
                        return orders, croissants_orders, jams_orders

                    orders.append(Order("PICNIC_BASKET2", best_bid, -1 * execute_volume))
                    self.internal_position["PICNIC_BASKET2"] -= execute_volume

                    croissants_order, jams_order = self.convert_synthetic2_orders(synthetic_order_depth,
                                                                                  state,
                                                                                  Order("SYNTHETIC2", synthetic_ask,
                                                                                        execute_volume),
                                                                                  )

                    croissants_orders.append(croissants_order)
                    jams_orders.append(jams_order)

                    order_depth.buy_orders[best_bid] -= execute_volume
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]

                    state.order_depths["CROISSANTS"].sell_orders[croissants_order.price] += (4 * execute_volume)
                    if state.order_depths["CROISSANTS"].sell_orders[croissants_order.price] == 0:
                        del state.order_depths["CROISSANTS"].sell_orders[croissants_order.price]

                    state.order_depths["JAMS"].sell_orders[jams_order.price] += (2 * execute_volume)
                    if state.order_depths["JAMS"].sell_orders[jams_order.price] == 0:
                        del state.order_depths["JAMS"].sell_orders[jams_order.price]

                    sell_order_volume += execute_volume

            # z score too low, picnic undervalued, buy picnic, sell synthetic
            elif z_score < -1 * z_score_threshold:
                if position < limit:
                    target_quantity = limit - position

                    try:
                        best_ask = min(order_depth.sell_orders.keys())
                    except:
                        return orders, croissants_orders, jams_orders

                    best_ask_volume = -1 * order_depth.sell_orders[best_ask]

                    synthetic_bid_volume = synthetic_order_depth.buy_orders[synthetic_bid]

                    orderbook_volume = min(best_ask_volume, synthetic_bid_volume)
                    execute_volume = min(orderbook_volume, target_quantity)

                    if execute_volume == 0:
                        return orders, croissants_orders, jams_orders

                    orders.append(Order("PICNIC_BASKET2", best_ask, execute_volume))
                    self.internal_position["PICNIC_BASKET2"] += execute_volume

                    croissants_order, jams_order = self.convert_synthetic2_orders(synthetic_order_depth,
                                                                                  state,
                                                                                  Order("SYNTHETIC2",
                                                                                        synthetic_bid,
                                                                                        -1 * execute_volume),
                                                                                  )

                    croissants_orders.append(croissants_order)
                    jams_orders.append(jams_order)

                    order_depth.sell_orders[best_ask] += execute_volume
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]

                    state.order_depths["CROISSANTS"].buy_orders[croissants_order.price] -= (4 * execute_volume)
                    if state.order_depths["CROISSANTS"].buy_orders[croissants_order.price] == 0:
                        del state.order_depths["CROISSANTS"].buy_orders[croissants_order.price]

                    state.order_depths["JAMS"].buy_orders[jams_order.price] -= (2 * execute_volume)
                    if state.order_depths["JAMS"].buy_orders[jams_order.price] == 0:
                        del state.order_depths["JAMS"].buy_orders[jams_order.price]

                    buy_order_volume += execute_volume

        return orders, croissants_orders, jams_orders

    def convert_synthetic3_orders(self, synthetic_order_depth, state: TradingState, synthetic_order):
        best_bid = max(synthetic_order_depth.buy_orders.keys())
        best_ask = min(synthetic_order_depth.sell_orders.keys())

        price = synthetic_order.price
        quantity = synthetic_order.quantity

        order_depths = state.order_depths

        # buying synthetic
        if quantity > 0:
            picnic1_price = min(order_depths["PICNIC_BASKET1"].sell_orders.keys())
            picnic2_price = max(order_depths["PICNIC_BASKET2"].buy_orders.keys())

        # selling synthetic
        elif quantity < 0:
            picnic1_price = max(order_depths["PICNIC_BASKET1"].buy_orders.keys())
            picnic2_price = min(order_depths["PICNIC_BASKET2"].sell_orders.keys())

        picnic1_order = Order("PICNIC_BASKET1", picnic1_price, quantity)
        self.internal_position["PICNIC_BASKET1"] += quantity

        if quantity >= 0:
            picnic2_order = Order("PICNIC_BASKET2", picnic2_price, -1 * int(quantity * (3 / 2)))
            self.internal_position["PICNIC_BASKET2"] -= int(quantity * (3 / 2))
        else:
            picnic2_order = Order("PICNIC_BASKET2", picnic2_price, int(-1 * quantity * (3 / 2)))
            self.internal_position["PICNIC_BASKET2"] += int(-1 * quantity * (3 / 2))

        return picnic1_order, picnic2_order

    # get the order depth of the synthetic2
    def synthetic3_order_depth(self, state: TradingState):
        order_depths = state.order_depths
        synthetic_order_depth = OrderDepth()

        picnic1_best_ask = min(order_depths["PICNIC_BASKET1"].sell_orders.keys())
        picnic1_best_bid = max(order_depths["PICNIC_BASKET1"].buy_orders.keys())

        picnic2_best_ask = min(order_depths["PICNIC_BASKET2"].sell_orders.keys())
        picnic2_best_bid = max(order_depths["PICNIC_BASKET2"].buy_orders.keys())

        synthetic_best_ask = picnic1_best_ask + int(-1 * picnic2_best_bid * (3 / 2))
        synthetic_best_bid = picnic1_best_bid - int(picnic2_best_ask * (3 / 2))

        picnic1_ask_volume = -1 * order_depths["PICNIC_BASKET1"].sell_orders[picnic1_best_ask]
        picnic1_bid_volume = order_depths["PICNIC_BASKET1"].buy_orders[picnic1_best_bid]
        try:
            picnic1_position = self.internal_position["PICNIC_BASKET1"]
        except:
            picnic1_position = 0

        picnic2_ask_volume = -1 * order_depths["PICNIC_BASKET2"].sell_orders[picnic2_best_ask]
        picnic2_bid_volume = order_depths["PICNIC_BASKET2"].buy_orders[picnic2_best_bid]
        try:
            picnic2_position = self.internal_position["PICNIC_BASKET2"]
        except:
            picnic2_position = 0

        synthetic_ask_volume = min(int(picnic1_ask_volume), int(60 - picnic1_position),
                                   int(picnic2_bid_volume / (3 / 2)), int((100 + picnic2_position) / (3 / 2)),
                                   )
        synthetic_bid_volume = min(int(picnic1_bid_volume), int(60 + picnic1_position),
                                   int(picnic2_ask_volume / (3 / 2)), int((100 - picnic2_position) / (3 / 2)),
                                   )

        synthetic_order_depth.sell_orders[synthetic_best_ask] = -1 * synthetic_ask_volume
        synthetic_order_depth.buy_orders[synthetic_best_bid] = synthetic_bid_volume

        return synthetic_order_depth

    def djembes_strategy(self, state: TradingState, limit: int, stored_data, first: bool):
        orders: List[Order] = []
        picnic1_orders: List[Order] = []
        picnic2_orders: List[Order] = []

        order_depth = state.order_depths["DJEMBES"]
        position = self.internal_position["DJEMBES"] if "DJEMBES" in self.internal_position else 0
        # position = state.position["DJEMBES"] if "DJEMBES" in state.position else 0

        djembes_fair_value = stored_data["DJEMBES"]["fair_value"]

        # "fair" value for synthetic
        synthetic_order_depth = self.synthetic3_order_depth(state)

        synthetic_ask = max(synthetic_order_depth.sell_orders.keys())
        synthetic_bid = min(synthetic_order_depth.buy_orders.keys())

        synthetic_fair_value = (synthetic_ask + synthetic_bid) / 2

        take_width = 1
        buy_order_volume = 0
        sell_order_volume = 0

        # z score for spread based on rolling window  30/10 220402
        z_trade = True
        window = 30
        z_score_threshold = 20

        spread = djembes_fair_value - synthetic_fair_value

        if first:
            stored_data["DJEMBES"]["spread_history"].append(spread)

        if len(stored_data["DJEMBES"]["spread_history"]) < 5:
            z_trade = False
        elif len(stored_data["DJEMBES"]["spread_history"]) > window:
            stored_data["DJEMBES"]["spread_history"].pop(0)

        if z_trade:
            spread_std = np.std(stored_data["DJEMBES"]["spread_history"])

            if spread_std == 0:
                z_score = float("inf")
            else:
                z_score = (spread - 250.5064) / spread_std

            # if we are much higher, then djembes is overvalued and we should sell djembes, buy synthetic
            if z_score > z_score_threshold:
                if position > -1 * limit:
                    target_quantity = position + limit

                    try:
                        best_bid = max(order_depth.buy_orders.keys())
                    except:
                        return orders, picnic1_orders, picnic2_orders

                    best_bid_volume = order_depth.buy_orders[best_bid]

                    synthetic_ask_volume = -1 * synthetic_order_depth.sell_orders[synthetic_ask]

                    orderbook_volume = min(best_bid_volume, synthetic_ask_volume)
                    execute_volume = min(orderbook_volume, target_quantity)

                    if execute_volume == 0:
                        return orders, picnic1_orders, picnic2_orders

                    orders.append(Order("DJEMBES", best_bid, -1 * execute_volume))
                    self.internal_position["DJEMBES"] -= execute_volume

                    picnic1_order, picnic2_order = self.convert_synthetic3_orders(synthetic_order_depth,
                                                                                  state,
                                                                                  Order("SYNTHETIC3", synthetic_ask,
                                                                                        execute_volume),
                                                                                  )

                    picnic1_orders.append(picnic1_order)
                    picnic2_orders.append(picnic2_order)

                    order_depth.buy_orders[best_bid] -= execute_volume
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]

                    state.order_depths["PICNIC_BASKET1"].sell_orders[picnic1_order.price] += execute_volume
                    if state.order_depths["PICNIC_BASKET1"].sell_orders[picnic1_order.price] == 0:
                        del state.order_depths["PICNIC_BASKET1"].sell_orders[picnic1_order.price]

                    state.order_depths["PICNIC_BASKET2"].buy_orders[picnic2_order.price] -= int(
                        execute_volume * (3 / 2))
                    if state.order_depths["PICNIC_BASKET2"].buy_orders[picnic2_order.price] == 0:
                        del state.order_depths["PICNIC_BASKET2"].buy_orders[picnic2_order.price]

                    sell_order_volume += execute_volume

            # z score too low, picnic undervalued, buy picnic, sell synthetic
            elif z_score < -1 * z_score_threshold:
                if position < limit:
                    target_quantity = limit - position

                    try:
                        best_ask = min(order_depth.sell_orders.keys())
                    except:
                        return orders, picnic1_orders, picnic2_orders

                    best_ask_volume = -1 * order_depth.sell_orders[best_ask]

                    synthetic_bid_volume = synthetic_order_depth.buy_orders[synthetic_bid]

                    orderbook_volume = min(best_ask_volume, synthetic_bid_volume)
                    execute_volume = min(orderbook_volume, target_quantity)

                    if execute_volume == 0:
                        return orders, picnic1_orders, picnic2_orders

                    orders.append(Order("DJEMBES", best_ask, execute_volume))
                    self.internal_position["DJEMBES"] += execute_volume

                    picnic1_order, picnic2_order = self.convert_synthetic3_orders(synthetic_order_depth,
                                                                                  state,
                                                                                  Order("SYNTHETIC3",
                                                                                        synthetic_bid,
                                                                                        -1 * execute_volume),
                                                                                  )

                    picnic1_orders.append(picnic1_order)
                    picnic2_orders.append(picnic2_order)

                    order_depth.sell_orders[best_ask] += execute_volume
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]

                    state.order_depths["PICNIC_BASKET1"].buy_orders[picnic1_order.price] -= execute_volume
                    if state.order_depths["PICNIC_BASKET1"].buy_orders[picnic1_order.price] == 0:
                        del state.order_depths["PICNIC_BASKET1"].buy_orders[picnic1_order.price]

                    state.order_depths["PICNIC_BASKET2"].sell_orders[picnic2_order.price] += int(
                        execute_volume * (3 / 2))
                    if state.order_depths["PICNIC_BASKET2"].sell_orders[picnic2_order.price] == 0:
                        del state.order_depths["PICNIC_BASKET2"].sell_orders[picnic2_order.price]

                    buy_order_volume += execute_volume

        return orders, picnic1_orders, picnic2_orders

    def spread_picker(self,
                      state: TradingState,
                      arb1_exists: bool,
                      arb2_exists: bool,
                      arb3_exists: bool,
                      stored_data,
                      ):
        if arb1_exists:
            picnic1_order_depth = state.order_depths["PICNIC_BASKET1"]

            try:
                mm_ask = max(picnic1_order_depth.sell_orders.keys())
                mm_bid = min(picnic1_order_depth.buy_orders.keys())
                picnic1_fair_value = (mm_ask + mm_bid) / 2
                stored_data["PICNIC_BASKET1"]["fair_value"] = picnic1_fair_value

                synthetic1_order_depth = self.synthetic1_order_depth(state)
                synthetic1_ask = max(synthetic1_order_depth.sell_orders.keys())
                synthetic1_bid = min(synthetic1_order_depth.buy_orders.keys())
                synthetic1_fair_value = (synthetic1_ask + synthetic1_bid) / 2

                spread1 = picnic1_fair_value - synthetic1_fair_value + 105.748375
            except:
                spread1 = 0
        else:
            spread1 = 0

        if arb2_exists:
            picnic2_order_depth = state.order_depths["PICNIC_BASKET2"]

            try:
                mm_ask = max(picnic2_order_depth.sell_orders.keys())
                mm_bid = min(picnic2_order_depth.buy_orders.keys())
                picnic2_fair_value = (mm_ask + mm_bid) / 2
                stored_data["PICNIC_BASKET2"]["fair_value"] = picnic2_fair_value

                synthetic2_order_depth = self.synthetic2_order_depth(state)
                synthetic2_ask = max(synthetic2_order_depth.sell_orders.keys())
                synthetic2_bid = min(synthetic2_order_depth.buy_orders.keys())
                synthetic2_fair_value = (synthetic2_ask + synthetic2_bid) / 2

                spread2 = picnic2_fair_value - synthetic2_fair_value - 96.50535
            except:
                spread2 = 0
        else:
            spread2 = 0

        if arb3_exists:
            djembes_order_depth = state.order_depths["DJEMBES"]

            try:
                mm_ask = max(djembes_order_depth.sell_orders.keys())
                mm_bid = min(djembes_order_depth.buy_orders.keys())
                djembes_fair_value = (mm_ask + mm_bid) / 2
                stored_data["DJEMBES"]["fair_value"] = djembes_fair_value

                synthetic3_order_depth = self.synthetic3_order_depth(state)
                synthetic3_ask = max(synthetic3_order_depth.sell_orders.keys())
                synthetic3_bid = min(synthetic3_order_depth.buy_orders.keys())
                synthetic3_fair_value = (synthetic3_ask + synthetic3_bid) / 2

                spread3 = djembes_fair_value - synthetic3_fair_value - 250.5064
            except:
                spread3 = 0
        else:
            spread3 = 0

        if spread1 != 0 and abs(spread1) == max(abs(spread1), abs(spread2), abs(spread3)):
            best = "PICNIC_BASKET1"
        elif spread2 != 0 and abs(spread2) == max(abs(spread1), abs(spread2), abs(spread3)):
            best = "PICNIC_BASKET2"
        elif spread3 != 0 and abs(spread3) == max(abs(spread1), abs(spread2), abs(spread3)):
            best = "DJEMBES"
        else:
            best = None

        return best
    
    def picnic_olivia_helper(self, state: TradingState, limit: int, asset: str):
        orders: List[Order] = []
        
        order_depth = state.order_depths[asset]
        position = state.position[asset] if asset in state.position else 0
        trades = state.market_trades.get(asset, [])
        trades = [t for t in trades if t.timestamp == state.timestamp-100]   
            
        signal = self.picnic_olivia[asset]
        
        if signal == "buy":
            if position == limit:
                self.picnic_olivia[asset] = "trade"
                return orders
            
            for ask_price in sorted(order_depth.sell_orders.keys()):
                best_ask_amount = -1 * order_depth.sell_orders[ask_price] 
                quantity = min(best_ask_amount, limit - position)
                orders.append(Order(asset, ask_price, quantity))

                position += quantity
                if quantity == 0:
                    break

        elif signal == "sell":
            if position <= -limit:
                self.picnic_olivia[asset] = "trade"
                return orders
            
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                best_bid_amount = order_depth.buy_orders[bid_price]
                quantity = min(best_bid_amount, limit + position)
                orders.append(Order(asset, bid_price, -1 * quantity))

                position -= quantity
                
                if quantity == 0:
                    break
            
        return orders

    def correlated_picnic_olivia_helper(self, state: TradingState, limit: int, asset: str):
        orders: List[Order] = []
        
        order_depth = state.order_depths[asset]
        position = state.position[asset] if asset in state.position else 0
        trades = state.market_trades.get(asset, [])
        trades = [t for t in trades if t.timestamp == state.timestamp-100]   
            
        signal = self.picnic_olivia[asset]
        
        if signal == "buy":
            if position == limit:
                self.picnic_olivia[asset] = "trade"
                return orders
            
            for ask_price in sorted(order_depth.sell_orders.keys()):
                best_ask_amount = -1 * order_depth.sell_orders[ask_price] 
                quantity = min(best_ask_amount, limit - position)
                orders.append(Order(asset, ask_price, quantity))

                position += quantity
                if quantity == 0:
                    break
        
        elif signal == "buy_second":
            if position >= 0:
                self.picnic_olivia[asset] = "done"
                return orders
            
            for ask_price in sorted(order_depth.sell_orders.keys()):
                best_ask_amount = -1 * order_depth.sell_orders[ask_price]
                if position < 0: 
                    quantity = min(best_ask_amount, limit - position, -position)
                else:
                    quantity = min(best_ask_amount, limit - position)
                orders.append(Order(asset, ask_price, quantity))

                position += quantity
                if quantity == 0:
                    break 

        elif signal == "sell":
            if position <= -limit:
                self.picnic_olivia[asset] = "trade"
                return orders
            
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                best_bid_amount = order_depth.buy_orders[bid_price]
                quantity = min(best_bid_amount, limit + position)
                orders.append(Order(asset, bid_price, -1 * quantity))

                position -= quantity
                
                if quantity == 0:
                    break
        
        elif signal == "sell_second":
            if position <= 0:
                self.picnic_olivia[asset] = "done"
                return orders
            
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                best_bid_amount = order_depth.buy_orders[bid_price]
                if position > 0:
                    quantity = min(best_bid_amount, limit + position, position)
                else:
                    quantity = min(best_bid_amount, limit + position)
                orders.append(Order(asset, bid_price, -1 * quantity))

                position -= quantity
                
                if quantity == 0:
                    break
            
        return orders
    
    def picnic_olivia_strategy(self, state: TradingState, croissants_limit: int, picnic1_limit: int, picnic2_limit: int):
        trades = state.market_trades.get("CROISSANTS", [])
        trades = [t for t in trades if t.timestamp == state.timestamp-100]   
  
        if any(t.buyer == "Olivia" for t in trades):
            self.resume_picnic_arb = False
            if self.picnic_olivia["CROISSANTS"] == "trade":
                self.picnic_olivia["PICNIC_BASKET1"] = "buy_second"
                self.picnic_olivia["PICNIC_BASKET2"] = "buy_second"
            else:
                self.picnic_olivia["PICNIC_BASKET1"] = "buy"
                self.picnic_olivia["PICNIC_BASKET2"] = "buy"
            self.picnic_olivia["CROISSANTS"] = "buy"
            
            
        if any(t.seller == "Olivia" for t in trades):
            self.resume_picnic_arb = False
            if self.picnic_olivia["CROISSANTS"] == "trade":
                self.picnic_olivia["PICNIC_BASKET1"] = "sell_second"
                self.picnic_olivia["PICNIC_BASKET2"] = "sell_second"
            else:
                self.picnic_olivia["PICNIC_BASKET1"] = "sell"
                self.picnic_olivia["PICNIC_BASKET2"] = "sell"
            self.picnic_olivia["CROISSANTS"] = "sell"
            
        croissants_orders = self.picnic_olivia_helper(state, croissants_limit, "CROISSANTS")
        picnic1_orders = self.correlated_picnic_olivia_helper(state, picnic1_limit, "PICNIC_BASKET1")
        picnic2_orders = self.correlated_picnic_olivia_helper(state, picnic2_limit, "PICNIC_BASKET2")
            
        if croissants_orders == [] and picnic1_orders == [] and picnic2_orders == []:
            self.resume_picnic_arb = True
            
        return croissants_orders, picnic1_orders, picnic2_orders
        
    def add_to_dict_list(self, d, key, values):
        if key in d and isinstance(d[key], list):
            d[key].extend(values)
        else:
            d[key] = list(values)
            
    def run(self, state: TradingState):
        stored_data = json.loads(state.traderData) if state.traderData else {}

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
            if product not in stored_data:
                stored_data[product] = {
                    "spread_history": [],
                    "fair_value": 0,
                }

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
            # if product == "SQUID_INK":
            #     result["SQUID_INK"] = self.ink_strategy(state, POSITION_LIMITS["SQUID_INK"])
                    
            # if product == "PICNIC_BASKET1":
            #     croissants_orders, picnic1_orders, picnic2_orders = self.picnic_olivia_strategy(state, POSITION_LIMITS["CROISSANTS"], POSITION_LIMITS["PICNIC_BASKET1"], POSITION_LIMITS["PICNIC_BASKET2"])
                
            #     result["CROISSANTS"] = croissants_orders
            #     result["PICNIC_BASKET1"] = picnic1_orders
            #     result["PICNIC_BASKET2"] = picnic2_orders
                
            #     for p in ["CROISSANTS",
            #             "JAMS",
            #             "DJEMBES",
            #             "PICNIC_BASKET1",
            #             "PICNIC_BASKET2"]:
            #         self.internal_position[p] = state.position[p] if p in state.position else 0
                
            #     if self.resume_picnic_arb == True:
            #         arb1_exists = True
            #         arb2_exists = True
            #         arb3_exists = True
            #         picnic1_first = True
            #         picnic2_first = True
            #         djembes_first = True
                    
            #         while arb1_exists or arb2_exists or arb3_exists:
            #             best = self.spread_picker(state,
            #                                     arb1_exists,
            #                                     arb2_exists,
            #                                     arb3_exists,
            #                                     stored_data,
            #                                     )
            #             if best == "PICNIC_BASKET1":
            #                 orders, croissants_orders, jams_orders, djembes_orders = self.picnic1_strategy(state, POSITION_LIMITS["PICNIC_BASKET1"], stored_data, picnic1_first)
            #                 self.add_to_dict_list(result, "PICNIC_BASKET1", orders)
            #                 self.add_to_dict_list(result, "CROISSANTS", croissants_orders)
            #                 self.add_to_dict_list(result, "JAMS", jams_orders)
            #                 self.add_to_dict_list(result, "DJEMBES", djembes_orders)
            #                 picnic1_first = False
            #                 if orders == [] or orders[0].quantity == 0:
            #                     arb1_exists = False
            #             elif best == "PICNIC_BASKET2":
            #                 orders, croissants_orders, jams_orders = self.picnic2_strategy(state, POSITION_LIMITS[
            #                     "PICNIC_BASKET2"], stored_data, picnic2_first)
            #                 self.add_to_dict_list(result, "PICNIC_BASKET2", orders)
            #                 self.add_to_dict_list(result, "CROISSANTS", croissants_orders)
            #                 self.add_to_dict_list(result, "JAMS", jams_orders)
            #                 picnic2_first = False
            #                 if orders == [] or orders[0].quantity == 0:
            #                     arb2_exists = False
            #             elif best == "DJEMBES":
            #                 orders, picnic1_orders, picnic2_orders = self.djembes_strategy(state, POSITION_LIMITS["DJEMBES"], stored_data, djembes_first)
            #                 self.add_to_dict_list(result, "DJEMBES", orders)
            #                 self.add_to_dict_list(result, "PICNIC_BASKET1", picnic1_orders)
            #                 self.add_to_dict_list(result, "PICNIC_BASKET2", picnic2_orders)  
            #                 djembes_first = False
            #                 if orders == [] or orders[0].quantity == 0:
            #                     arb3_exists = False
            #             else:
            #                 break
                
                
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
