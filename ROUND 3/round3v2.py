import json
from typing import Any, Dict, List, Tuple, Optional
import math
import numpy as np

from datamodel import (
    Order, OrderDepth, TradingState, Trade, Symbol, ProsperityEncoder, Listing, Observation # Make sure all needed imports are here
)

# Logger Class (Using the robust version from previous examples)
class Logger:
    def __init__(self) -> None: self.logs = ""; self.max_log_length = 3750
    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        # (Implementation from previous step)
        msg = sep.join(map(str, objects)) + end
        if len(self.logs) + len(msg) > self.max_log_length:
            space_left = self.max_log_length - len(self.logs)
            if space_left > 10: msg = msg[:space_left - 3] + "...\n"
            else: msg = ""
        self.logs += msg
    def flush(self, state: TradingState, orders: Dict[Symbol, List[Order]], conversions: int, trader_data: str) -> None:
        # (Implementation from previous step - including compression methods)
        try:
            # Ensure compress methods exist if reusing logger
            output_list = [self.compress_state(state, trader_data), self.compress_orders(orders), conversions, "", self.logs]
            output_str = json.dumps(output_list, cls=ProsperityEncoder, separators=(",", ":"))
            print(output_str)
        except Exception as e:
            fallback = {"error": f"Logger flush error: {e}", "timestamp": getattr(state, "timestamp", "N/A")}
            print(json.dumps(fallback))
        finally: self.logs = ""
    # --- Compression Methods (Assume they exist here) ---
    def compress_state(self, state: TradingState, trader_data: str) -> List[Any]:
        # Compresses the TradingState for logging/output
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
        observations = self.compress_observations(getattr(state, 'observations', None)) # Pass observations object

        return [state.timestamp, trader_data, listings_compressed, order_depths_compressed, own_trades, market_trades, position, observations]

    def compress_trades(self, trades: Dict[Symbol, List[Trade]]) -> List[List[Any]]:
        compressed = []
        if trades:
             for trade_list in trades.values():
                  for t in trade_list:
                      compressed.append([
                           getattr(t, 'symbol', None), getattr(t, 'price', None), getattr(t, 'quantity', None),
                           getattr(t, 'buyer', None), getattr(t, 'seller', None), getattr(t, 'timestamp', None)
                      ])
        return compressed

    def compress_orders(self, orders: Dict[Symbol, List[Order]]) -> List[List[Any]]:
        compressed = []
        if orders:
             for sym, order_list in orders.items():
                  for o in order_list:
                      compressed.append([getattr(o, 'symbol', None), getattr(o, 'price', None), getattr(o, 'quantity', None)])
        return compressed

    def compress_observations(self, obs: Optional[Observation]) -> List[Any]:
        # Processes observation data for logging
        if not obs: return [{}, {}]
        conv_obs, plain_obs = {}, {}
        try:
             if hasattr(obs, 'conversionObservations') and obs.conversionObservations:
                  expected_conv_attrs = ['bidPrice', 'askPrice', 'transportFees', 'exportTariff', 'importTariff', 'sunlight', 'humidity'] # Verify these attrs
                  for product, data in obs.conversionObservations.items():
                       if data is not None: conv_obs[product] = [getattr(data, attr, None) for attr in expected_conv_attrs]
                       else: conv_obs[product] = [None] * len(expected_conv_attrs)
        except Exception as e: print(f"ERROR compressing conversionObservations: {e}"); conv_obs["error"] = str(e)
        try:
             if hasattr(obs, 'plainValueObservations') and obs.plainValueObservations: plain_obs = dict(obs.plainValueObservations)
        except Exception as e: print(f"ERROR compressing plainValueObservations: {e}"); plain_obs["error"] = str(e)
        return [plain_obs, conv_obs]


logger = Logger()

class Trader:
    # Define products and their limits - ENSURE THESE ARE CORRECT FOR THE ROUND
    PRODUCTS = {
        "VOLCANIC_ROCK": 400,
        "VOLCANIC_ROCK_VOUCHER_9500": 200,
        "VOLCANIC_ROCK_VOUCHER_9750": 200,
        "VOLCANIC_ROCK_VOUCHER_10000": 200,
        "VOLCANIC_ROCK_VOUCHER_10250": 200,
        "VOLCANIC_ROCK_VOUCHER_10500": 200,
    }
    # Parameters for the Z-score strategy (same for all products as requested)
    ZSCORE_WINDOW = 100
    ZSCORE_THRESHOLD = 1.95

    # Key for persistent data storage for this strategy
    STATE_DATA_KEY = "zscore_strategy_data_v4" # Use a distinct key

    # --- Z-Score Strategy Function (Generalized) ---
    def _run_zscore_strategy(
        self,
        symbol: Symbol,         # Product symbol (e.g., "VOLCANIC_ROCK")
        limit: int,             # Position limit for this specific symbol
        state: TradingState,
        stored_data: Dict,      # Expects the strategy-specific part of trader_data_map for THIS symbol
    ) -> List[Order]:
        """Implements the Z-score strategy based on worst bid/ask fair value for a given symbol."""
        orders: List[Order] = []

        order_depth = state.order_depths.get(symbol)
        if not order_depth:
            # logger.print(f"No order depth found for {symbol}.")
            return orders

        # History is stored per symbol within stored_data
        spread_history = stored_data.setdefault("spread_history", [])
        position = state.position.get(symbol, 0)

        # Calculate fair value using WORST bid/ask
        try:
            if not order_depth.sell_orders or not order_depth.buy_orders:
                 # logger.print(f"Order depth missing bids or asks for {symbol}.")
                 return orders
            worst_ask = max(order_depth.sell_orders.keys())
            worst_bid = min(order_depth.buy_orders.keys())
            fair_value = (worst_ask + worst_bid) / 2
        except Exception as e:
            logger.print(f"Error calculating fair value for {symbol}: {e}")
            return orders

        # Update history
        spread_history.append(fair_value)
        if len(spread_history) > self.ZSCORE_WINDOW:
            spread_history.pop(0)

        # Calculate Z-Score
        if len(spread_history) < 5: # Min history requirement
            return orders

        try:
            mean = np.mean(spread_history)
            std = np.std(spread_history)
        except Exception as e:
             logger.print(f"Error calculating mean/std for {symbol}: {e}")
             return orders

        if std is None or std < 1e-6:
            return orders # Cannot calculate Z-score if std dev is zero

        z_score = (fair_value - mean) / std
        # logger.print(f"Symbol: {symbol}, Z-Score: {z_score:.3f} (Mean: {mean:.2f}, Std: {std:.2f}, FV: {fair_value:.2f})")


        # Trading Logic (Identical for all products)
        if z_score > self.ZSCORE_THRESHOLD:
            # Sell signal
            if order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders.keys())
                possible_volume = order_depth.buy_orders[best_bid]
                max_sell = max(0, limit + position) # Ensure non-negative
                volume = min(possible_volume, max_sell)
                if volume > 0:
                    logger.print(f"{symbol} Z-Score SELL: Z={z_score:.3f} > {self.ZSCORE_THRESHOLD}. Selling {volume} @ {best_bid}")
                    orders.append(Order(symbol, int(round(best_bid)), -int(round(volume))))
            # else: logger.print(f"{symbol} Z-Score SELL Signal: No buy orders.")


        elif z_score < -self.ZSCORE_THRESHOLD:
            # Buy signal
            if order_depth.sell_orders:
                best_ask = min(order_depth.sell_orders.keys())
                possible_volume = abs(order_depth.sell_orders[best_ask])
                max_buy = max(0, limit - position) # Ensure non-negative
                volume = min(possible_volume, max_buy)
                if volume > 0:
                    logger.print(f"{symbol} Z-Score BUY: Z={z_score:.3f} < {-self.ZSCORE_THRESHOLD}. Buying {volume} @ {best_ask}")
                    orders.append(Order(symbol, int(round(best_ask)), int(round(volume))))
            # else: logger.print(f"{symbol} Z-Score BUY Signal: No sell orders.")

        return orders

    # --- Main run method ---
    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        """
        Main trading method. Loads state, runs the Z-score strategy
        for VOLCANIC_ROCK and all VOUCHERS, returns combined orders.
        """
        result: Dict[Symbol, List[Order]] = {}
        conversions = 0
        trader_data_map = {}

        # Load state
        try:
            if state.traderData: trader_data_map = json.loads(state.traderData)
        except json.JSONDecodeError: logger.print("Error decoding traderData JSON.")

        # Get or initialize the persistent data structure for this strategy
        # Data for each symbol will be stored under its key within this dict
        strategy_persistent_data = trader_data_map.setdefault(self.STATE_DATA_KEY, {})

        # Iterate through all products defined with their limits
        for symbol, limit in self.PRODUCTS.items():
            # Get the specific data storage for this symbol (e.g., its history)
            # Creates dict for symbol if it doesn't exist
            symbol_data = strategy_persistent_data.setdefault(symbol, {})

            # Run the Z-score strategy for the current symbol
            symbol_orders = self._run_zscore_strategy(symbol, limit, state, symbol_data)

            # Add generated orders to the result dictionary
            if symbol_orders:
                 result[symbol] = symbol_orders

        # --- Logic for other strategies or products could be added here ---

        # Prepare Trader Data and Return
        # strategy_persistent_data was modified in-place within trader_data_map
        trader_data_str = json.dumps(trader_data_map)

        logger.flush(state, result, conversions, trader_data_str)
        return result, conversions, trader_data_str