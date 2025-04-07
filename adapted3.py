import json
from abc import abstractmethod
from collections import deque
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, TypeAlias
from statistics import stdev


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

class Strategy:
    def __init__(self, symbol: str, config: dict) -> None:
        self.symbol = symbol
        # Configuration parameters for risk management and order placement
        self.position_limit = config.get("position_limit", 20)
        self.price_offset = config.get("price_offset", 1)  # Base price offset
        self.liquidation_window_size = config.get("liquidation_window_size", 10)
        self.volatility_multiplier = config.get("volatility_multiplier", 1)
        self.orders = []
    
    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> list:
        self.orders = []
        self.act(state)
        return self.orders

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    # Placeholder for order cancellation/adjustment logic
    def cancel_order(self, order: Order) -> None:
        logger.print(f"Cancelling order: {order}")

    def save(self) -> dict:
        return {}
    
    def load(self, data: dict) -> None:
        pass

class MarketMakingStrategy(Strategy):
    def __init__(self, symbol: str, config: dict) -> None:
        super().__init__(symbol, config)
        # Track whether the position is at the limit over a configurable window
        self.window = deque(maxlen=self.liquidation_window_size)
        # Keep recent true value estimates to calculate volatility dynamically.
        self.true_value_history = deque(maxlen=50)

    @abstractmethod
    def get_true_value(self, state: TradingState) -> int:
        raise NotImplementedError()

    def compute_volatility(self) -> float:
        """Compute volatility as the standard deviation of recent true value estimates."""
        if len(self.true_value_history) < 2:
            return 0
        return stdev(self.true_value_history)

    def get_popular_price(self, orders: dict, is_buy: bool = True) -> int:
        """
        Calculate a volume-weighted average price from an order book dictionary.
        For buy orders, orders are expected as price -> volume.
        """
        if not orders:
            return None
        total_volume = sum(abs(v) for v in orders.values())
        if total_volume == 0:
            return None
        weighted_sum = sum(price * abs(volume) for price, volume in orders.items())
        return round(weighted_sum / total_volume)

    def act(self, state: TradingState) -> None:
        # Determine the current true value and update history
        true_value = self.get_true_value(state)
        self.true_value_history.append(true_value)
        volatility = self.compute_volatility()
        # Adjust the price offset dynamically based on volatility
        dynamic_offset = max(self.price_offset, int(volatility * self.volatility_multiplier))
        
        order_depth = state.order_depths[self.symbol]
        # Sort orders: buy orders in descending order (best bid first), sell orders in ascending order (best ask first)
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        position = state.position.get(self.symbol, 0)
        to_buy = self.position_limit - position
        to_sell = self.position_limit + position

        # Update the window to track if the position has hit its limit
        self.window.append(abs(position) >= self.position_limit)
        soft_liquidate = (len(self.window) == self.liquidation_window_size and 
                          sum(self.window) >= self.liquidation_window_size / 2 and 
                          self.window[-1])
        hard_liquidate = (len(self.window) == self.liquidation_window_size and 
                          all(self.window))

        # Set dynamic pricing thresholds based on current position and dynamic offset
        max_buy_price = true_value - dynamic_offset if position > self.position_limit * 0.5 else true_value
        min_sell_price = true_value + dynamic_offset if position < -self.position_limit * 0.5 else true_value

        # Process sell orders to buy from the market (fill our buy orders)
        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity

        # Liquidation logic to reduce risk when near or over the limit
        if to_buy > 0 and hard_liquidate:
            quantity = to_buy // 2
            self.buy(true_value, quantity)
            to_buy -= quantity

        if to_buy > 0 and soft_liquidate:
            quantity = to_buy // 2
            self.buy(true_value - dynamic_offset, quantity)
            to_buy -= quantity

        # If orders remain, use volume-weighted popular price from the buy side
        if to_buy > 0:
            popular_buy_price = self.get_popular_price(dict(buy_orders), is_buy=True)
            price = min(max_buy_price, (popular_buy_price + dynamic_offset) if popular_buy_price else max_buy_price)
            self.buy(price, to_buy)

        # Process buy orders to sell from the market (fill our sell orders)
        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity

        if to_sell > 0 and hard_liquidate:
            quantity = to_sell // 2
            self.sell(true_value, quantity)
            to_sell -= quantity

        if to_sell > 0 and soft_liquidate:
            quantity = to_sell // 2
            self.sell(true_value + dynamic_offset, quantity)
            to_sell -= quantity

        if to_sell > 0:
            popular_sell_price = self.get_popular_price(dict(sell_orders), is_buy=False)
            price = max(min_sell_price, (popular_sell_price - dynamic_offset) if popular_sell_price else min_sell_price)
            self.sell(price, to_sell)

    def save(self) -> dict:
        return {
            "window": list(self.window),
            "true_value_history": list(self.true_value_history)
        }

    def load(self, data: dict) -> None:
        if data is None:
            return
        self.window = deque(data.get("window", []), maxlen=self.liquidation_window_size)
        self.true_value_history = deque(data.get("true_value_history", []), maxlen=50)

# ---------------------------------------
# Specific Strategies Implementations
# ---------------------------------------

class ResinStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        """
        Compute a true value as the mid-price of the best bid and ask.
        If either side is missing, fall back to a default.
        """
        order_depth = state.order_depths[self.symbol]
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2
            return round(mid_price)
        return 10000  # Fallback value if order book data is incomplete

class KelpStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        """
        Compute the true value using the volume-weighted average of the best buy and sell orders.
        """
        order_depth = state.order_depths[self.symbol]
        popular_buy_price = self.get_popular_price(order_depth.buy_orders, is_buy=True)
        popular_sell_price = self.get_popular_price(order_depth.sell_orders, is_buy=False)
        if popular_buy_price is not None and popular_sell_price is not None:
            return round((popular_buy_price + popular_sell_price) / 2)
        elif order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return round((best_bid + best_ask) / 2)
        return 10000  # Fallback value

# ---------------------------------------
# Trader Orchestrator
# ---------------------------------------

class Trader:
    def __init__(self, config: dict = None) -> None:
        # Global strategy configuration; users can override these defaults.
        default_config = {
            "position_limit": 20,
            "price_offset": 1,
            "liquidation_window_size": 10,
            "volatility_multiplier": 1,
        }
        self.config = default_config if config is None else {**default_config, **config}
        self.strategies = {
            "RAINFOREST_RESIN": ResinStrategy("RAINFOREST_RESIN", self.config),
            "KELP": KelpStrategy("KELP", self.config),
        }

    def run(self, state: TradingState) -> tuple:
        logger.print("Current positions:", state.position)
        conversions = 0

        old_trader_data = json.loads(state.traderData) if state.traderData else {}
        new_trader_data = {}

        orders = {}
        for symbol, strategy in self.strategies.items():
            if symbol in old_trader_data:
                strategy.load(old_trader_data.get(symbol))
            if symbol in state.order_depths:
                orders[symbol] = strategy.run(state)
            new_trader_data[symbol] = strategy.save()

        trader_data = json.dumps(new_trader_data, separators=(",", ":"))
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data
