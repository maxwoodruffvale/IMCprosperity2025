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

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {s: [od.buy_orders, od.sell_orders] for s, od in order_depths.items()}

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        return [
            [t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
            for trade_list in trades.values()
            for t in trade_list
        ]

    def compress_observations(self, observations: Observation) -> list[Any]:
        return [
            observations.plainValueObservations,
            {
                p: [
                    o.bidPrice,
                    o.askPrice,
                    o.transportFees,
                    o.exportTariff,
                    o.importTariff,
                    o.sugarPrice,
                    o.sunlightIndex,
                ] for p, o in observations.conversionObservations.items()
            }
        ]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        return [[o.symbol, o.price, o.quantity] for order_list in orders.values() for o in order_list]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[:max_length - 3] + "..."

logger = Logger()

class Trader:
    def run(self, state: TradingState):
        stored_data = json.loads(state.traderData) if state.traderData else {}
        for product in ["RAINFOREST_RESIN", "KELP", "SQUID_INK"]:
            stored_data.setdefault(product, {})
            stored_data[product].setdefault("mins", [])
            stored_data[product].setdefault("maxs", [])
            stored_data[product].setdefault("mm_prices", [])
            stored_data[product].setdefault("momentum_override", (False, None))
            stored_data.setdefault("fair_history", {}).setdefault(product, [])

        POSITION_LIMITS = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
        }

        result = {}
        for product in POSITION_LIMITS:
            od = state.order_depths[product]
            position = state.position.get(product, 0)
            if od.sell_orders and od.buy_orders:
                best_ask = min(od.sell_orders.keys())
                best_bid = max(od.buy_orders.keys())
                mid = (best_ask + best_bid) / 2
                stored_data["fair_history"][product].append(mid)
                if len(stored_data["fair_history"][product]) > 20:
                    stored_data["fair_history"][product].pop(0)

        # Strategy calls with optimized versions
        result["RAINFOREST_RESIN"] = self.resin(state, POSITION_LIMITS["RAINFOREST_RESIN"], stored_data)
        result["KELP"] = self.kelp(state, POSITION_LIMITS["KELP"], stored_data)
        result["SQUID_INK"] = self.ink(state, POSITION_LIMITS["SQUID_INK"], stored_data)

        trader_data = json.dumps(stored_data, separators=(',', ':'))
        conversions = 0
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

    def resin(self, state, limit, stored_data):
        orders = []
        product = "RAINFOREST_RESIN"
        od = state.order_depths[product]
        position = state.position.get(product, 0)
        history = stored_data["fair_history"][product]

        if history:
            fair_value = sum(history) / len(history)
        else:
            fair_value = 10000

        volatility = np.std(history[-10:]) if len(history) > 5 else 0
        take_width = max(1, min(3, round(volatility * 0.5)))
        position_skew = 2 * (position / limit)
        bid = round(fair_value - take_width - position_skew)
        ask = round(fair_value + take_width - position_skew)

        if od.sell_orders:
            best_ask = min(od.sell_orders)
            quantity = min(-od.sell_orders[best_ask], limit - position)
            if best_ask <= fair_value - take_width and quantity > 0:
                orders.append(Order(product, best_ask, quantity))

        if od.buy_orders:
            best_bid = max(od.buy_orders)
            quantity = min(od.buy_orders[best_bid], limit + position)
            if best_bid >= fair_value + take_width and quantity > 0:
                orders.append(Order(product, best_bid, -quantity))

        remaining_buy = limit - position
        remaining_sell = limit + position
        if remaining_buy > 0:
            orders.append(Order(product, bid, remaining_buy))
        if remaining_sell > 0:
            orders.append(Order(product, ask, -remaining_sell))

        return orders

    def kelp(self, state, limit, stored_data):
        orders = []
        product = "KELP"
        od = state.order_depths[product]
        position = state.position.get(product, 0)
        fair_history = stored_data["fair_history"][product]

        if od.buy_orders and od.sell_orders:
            vwap_bid = sum(p * v for p, v in od.buy_orders.items()) / sum(od.buy_orders.values())
            vwap_ask = sum(p * abs(v) for p, v in od.sell_orders.items()) / sum(abs(v) for v in od.sell_orders.values())
            fair_value = (vwap_bid + vwap_ask) / 2
        else:
            return []

        alpha = 0.3
        threshold_price = alpha * fair_value + (1 - alpha) * fair_history[-1] if fair_history else fair_value
        if abs(threshold_price - fair_value) < 6:
            fair_history.append(fair_value)
            if len(fair_history) > 5:
                fair_history.pop(0)

        volatility = np.std(fair_history[-5:]) if len(fair_history) > 1 else 0
        take_width = max(1, min(3, round(volatility * 0.5)))
        position_factor = position / limit
        dynamic_spread = 2 * (1 + abs(position_factor))
        position_skew = 2 * position_factor

        if od.sell_orders:
            best_ask = min(od.sell_orders)
            price_improvement = threshold_price - best_ask
            scale = max(1, min(1.2, 1 + price_improvement / threshold_price))
            quantity = int(min(-od.sell_orders[best_ask], limit - position) * scale)
            if best_ask <= threshold_price - take_width and quantity > 0:
                orders.append(Order(product, best_ask, quantity))

        if od.buy_orders:
            best_bid = max(od.buy_orders)
            price_improvement = best_bid - threshold_price
            scale = max(1, min(1.2, 1 + price_improvement / threshold_price))
            quantity = int(min(od.buy_orders[best_bid], limit + position) * scale)
            if best_bid >= threshold_price + take_width and quantity > 0:
                orders.append(Order(product, best_bid, -quantity))

        bid = round(threshold_price - dynamic_spread - position_skew)
        ask = round(threshold_price + dynamic_spread - position_skew)

        if limit - position > 0:
            orders.append(Order(product, bid, limit - position))
        if limit + position > 0:
            orders.append(Order(product, ask, -limit - position))

        return orders

    def ink(self, state, limit, stored_data):
        orders = []
        product = "SQUID_INK"
        od = state.order_depths[product]
        position = state.position.get(product, 0)
        history = stored_data["fair_history"][product]

        if not od.sell_orders or not od.buy_orders:
            return []

        best_ask = min(od.sell_orders)
        best_bid = max(od.buy_orders)
        current_mid = (best_ask + best_bid) / 2
        history.append(current_mid)
        if len(history) > 100:
            history.pop(0)

        if len(history) >= 20:
            ar = sum(history[-5:]) / 5
            diff = history[-1] - history[-2] if len(history) > 1 else 0
            ma = sum(history[-i] - history[-i-1] for i in range(1, 4)) / 3 if len(history) >= 4 else 0
            predicted = 0.6 * ar + 0.2 * (history[-1] + diff) + 0.2 * (history[-1] + ma)
        else:
            predicted = current_mid

        fair_value = 0.7 * predicted + 0.3 * current_mid
        price_std = np.std(history[-10:]) if len(history) >= 10 else 0
        take_width = max(1, min(3, round(price_std * 0.67)))
        spread = max(1, take_width * 2)
        skew = -min(2, position / 10) if position > 0 else min(2, -position / 10)

        if best_ask <= fair_value - take_width:
            quantity = min(-od.sell_orders[best_ask], limit - position)
            if quantity > 0:
                orders.append(Order(product, best_ask, quantity))

        if best_bid >= fair_value + take_width:
            quantity = min(od.buy_orders[best_bid], limit + position)
            if quantity > 0:
                orders.append(Order(product, best_bid, -quantity))

        bid_price = round(fair_value - spread + skew)
        ask_price = round(fair_value + spread + skew)
        if limit - position > 0:
            orders.append(Order(product, bid_price, limit - position))
        if limit + position > 0:
            orders.append(Order(product, ask_price, -limit - position))

        return orders
