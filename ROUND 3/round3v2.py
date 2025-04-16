import json
from typing import Any, Dict, List, Tuple, Optional
import math
import numpy as np

from datamodel import (
    Order, OrderDepth, TradingState, Trade, Symbol, ProsperityEncoder, Listing, Observation
)

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750
    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        msg = sep.join(map(str, objects)) + end
        if len(self.logs) + len(msg) > self.max_log_length:
            space_left = self.max_log_length - len(self.logs)
            if space_left > 10:
                msg = msg[:space_left - 3] + "...\n"
            else:
                msg = ""
        self.logs += msg
    def flush(self, state: TradingState, orders: Dict[Symbol, List[Order]], conversions: int, trader_data: str) -> None:
        try:
            output_list = [self.compress_state(state, trader_data), self.compress_orders(orders), conversions, "", self.logs]
            output_str = json.dumps(output_list, cls=ProsperityEncoder, separators=(",", ":"))
            print(output_str)
        except Exception as e:
            fallback = {"error": f"Logger flush error: {e}", "timestamp": getattr(state, "timestamp", "N/A")}
            print(json.dumps(fallback))
        finally:
            self.logs = ""
    def compress_state(self, state: TradingState, trader_data: str) -> List[Any]:
        listings_compressed = []
        if hasattr(state, 'listings') and state.listings:
            for lst in state.listings.values():
                listings_compressed.append([lst.symbol, lst.product, lst.denomination])
        order_depths_compressed = {}
        if hasattr(state, 'order_depths') and state.order_depths:
            for sym, od in state.order_depths.items():
                order_depths_compressed[sym] = [list(od.buy_orders.items()), list(od.sell_orders.items())]
        own_trades = self.compress_trades(getattr(state, 'own_trades', {}))
        market_trades = self.compress_trades(getattr(state, 'market_trades', {}))
        position = getattr(state, 'position', {})
        observations = self.compress_observations(getattr(state, 'observations', None))
        return [state.timestamp, trader_data, listings_compressed, order_depths_compressed, own_trades, market_trades, position, observations]
    def compress_trades(self, trades: Dict[Symbol, List[Trade]]) -> List[List[Any]]:
        compressed = []
        for trade_list in trades.values():
            for t in trade_list:
                compressed.append([
                    getattr(t, 'symbol', None), getattr(t, 'price', None), getattr(t, 'quantity', None),
                    getattr(t, 'buyer', None), getattr(t, 'seller', None), getattr(t, 'timestamp', None)
                ])
        return compressed
    def compress_orders(self, orders: Dict[Symbol, List[Order]]) -> List[List[Any]]:
        compressed = []
        for sym, order_list in orders.items():
            for o in order_list:
                compressed.append([getattr(o, 'symbol', None), getattr(o, 'price', None), getattr(o, 'quantity', None)])
        return compressed
    def compress_observations(self, obs: Optional[Observation]) -> List[Any]:
        if not obs: return [{}, {}]
        conv_obs, plain_obs = {}, {}
        try:
            if hasattr(obs, 'conversionObservations') and obs.conversionObservations:
                expected_conv_attrs = ['bidPrice','askPrice','transportFees','exportTariff','importTariff','sunlight','humidity']
                for product, data in obs.conversionObservations.items():
                    if data is not None:
                        conv_obs[product] = [getattr(data, attr, None) for attr in expected_conv_attrs]
                    else:
                        conv_obs[product] = [None]*len(expected_conv_attrs)
        except Exception as e:
            print(f"ERROR compressing conversionObservations: {e}")
            conv_obs["error"] = str(e)
        try:
            if hasattr(obs, 'plainValueObservations') and obs.plainValueObservations:
                plain_obs = dict(obs.plainValueObservations)
        except Exception as e:
            print(f"ERROR compressing plainValueObservations: {e}")
            plain_obs["error"] = str(e)
        return [plain_obs, conv_obs]


logger = Logger()

class Trader:
    # Same PRODUCT limits
    PRODUCTS = {
        "VOLCANIC_ROCK": 400,
        "VOLCANIC_ROCK_VOUCHER_9500": 200,
        "VOLCANIC_ROCK_VOUCHER_9750": 200,
        "VOLCANIC_ROCK_VOUCHER_10000": 200,
        "VOLCANIC_ROCK_VOUCHER_10250": 200,
        "VOLCANIC_ROCK_VOUCHER_10500": 200,
    }
    ZSCORE_WINDOW = 95
    ZSCORE_THRESHOLD = 1.97

    # We'll allow a dynamic factor up to 3.0
    MAX_VOLUME_FACTOR = 3.0

    STATE_DATA_KEY = "zscore_strategy_data_v4"

    def _run_zscore_strategy(self, symbol: Symbol, limit: int, state: TradingState, stored_data: Dict) -> List[Order]:
        orders: List[Order] = []
        order_depth = state.order_depths.get(symbol)
        if not order_depth:
            return orders

        spread_history = stored_data.setdefault("spread_history", [])
        position = state.position.get(symbol, 0)

        if not order_depth.sell_orders or not order_depth.buy_orders:
            return orders

        worst_ask = max(order_depth.sell_orders.keys())
        worst_bid = min(order_depth.buy_orders.keys())
        fair_value = (worst_ask + worst_bid) / 2

        spread_history.append(fair_value)
        if len(spread_history) > self.ZSCORE_WINDOW:
            spread_history.pop(0)

        if len(spread_history) < 5:
            return orders

        mean = np.mean(spread_history)
        std = np.std(spread_history)
        if std < 1e-6:
            return orders

        z_score = (fair_value - mean) / std
        # logger.print(f"{symbol} zScore={z_score:.3f}, mean={mean:.2f}, std={std:.2f}, FV={fair_value:.2f}")

        if z_score > self.ZSCORE_THRESHOLD:
            # Sell signal
            best_bid = max(order_depth.buy_orders.keys())
            vol_bid = order_depth.buy_orders[best_bid]
            max_sell = max(0, limit + position)
            if max_sell > 0:
                # Dynamic scaling
                volume = self.scale_volume_based_on_zscore(z_score, self.ZSCORE_THRESHOLD, max_sell)
                final_volume = min(volume, vol_bid)
                if final_volume > 0:
                    logger.print(f"{symbol}: SELL z={z_score:.2f}, vol={final_volume} @ {best_bid}")
                    orders.append(Order(symbol, int(round(best_bid)), -int(round(final_volume))))

        elif z_score < -self.ZSCORE_THRESHOLD:
            # Buy signal
            best_ask = min(order_depth.sell_orders.keys())
            vol_ask = abs(order_depth.sell_orders[best_ask])
            max_buy = max(0, limit - position)
            if max_buy > 0:
                # Dynamic scaling
                # pass abs(z_score) for scaling
                volume = self.scale_volume_based_on_zscore(abs(z_score), self.ZSCORE_THRESHOLD, max_buy)
                final_volume = min(volume, vol_ask)
                if final_volume > 0:
                    logger.print(f"{symbol}: BUY z={z_score:.2f}, vol={final_volume} @ {best_ask}")
                    orders.append(Order(symbol, int(round(best_ask)), int(round(final_volume))))
        return orders

    def scale_volume_based_on_zscore(self, z_val: float, threshold: float, base_volume: int) -> int:
        """
        If z_val > threshold, overshoot = z_val - threshold, 
        we scale volume by ratio = 1 + overshoot/threshold, capped at MAX_VOLUME_FACTOR.
        e.g., if threshold=1.97, z_val=2.5 => overshoot=0.53 => ratio ~1.27 => trade ~ 1.27*base_volume
        """
        if z_val <= threshold:
            return base_volume

        overshoot = z_val - threshold
        ratio = 1.0 + overshoot / threshold
        ratio = min(ratio, self.MAX_VOLUME_FACTOR)
        scaled_vol = int(math.floor(base_volume * ratio))
        return scaled_vol

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        result: Dict[Symbol, List[Order]] = {}
        conversions = 0
        trader_data_map: Dict[str, Any] = {}

        # Load from state if any
        try:
            if state.traderData:
                trader_data_map = json.loads(state.traderData)
        except json.JSONDecodeError:
            logger.print("Error decoding traderData, starting fresh data.")

        strategy_persistent_data = trader_data_map.setdefault(self.STATE_DATA_KEY, {})

        # Run the z-score approach for each product
        for symbol, limit in self.PRODUCTS.items():
            symbol_data = strategy_persistent_data.setdefault(symbol, {})
            orders_for_sym = self._run_zscore_strategy(symbol, limit, state, symbol_data)
            if orders_for_sym:
                result[symbol] = orders_for_sym

        # Encode updated data
        trader_data_str = json.dumps(trader_data_map)
        logger.flush(state, result, conversions, trader_data_str)
        return result, conversions, trader_data_str
