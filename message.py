import json
import math
from typing import Any, List

# Required: these classes come from your environment or from the official datamodel.py
from datamodel import (
    Listing, Observation, Order, OrderDepth,
    ProsperityEncoder, Symbol, Trade, TradingState
)

###############################################################################
# Logger class (Keep your existing Logger class)
###############################################################################
class Logger:
    # ... (Logger class implementation remains the same) ...
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self,
              state: TradingState,
              orders: dict[Symbol, list[Order]],
              conversions: int,
              trader_data: str) -> None:
        # Truncate logs if necessary (same logic as before)
        base_json = self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ])
        base_length = len(base_json)
        max_item_length = (self.max_log_length - base_length) // 3
        if max_item_length < 0: max_item_length = 0 # Prevent negative length

        out_json = self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ])

        print(out_json)
        self.logs = "" # Reset logs after flushing

    # --- Keep all compress_* methods and to_json/truncate as they were ---
    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        # (Keep original implementation)
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
        # (Keep original implementation)
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        # (Keep original implementation)
        compressed = {}
        for symbol, od in order_depths.items():
            compressed[symbol] = [od.buy_orders, od.sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        # (Keep original implementation)
        compressed = []
        for arr in trades.values():
            for t in arr:
                compressed.append([
                    t.symbol,
                    t.price,
                    t.quantity,
                    t.buyer,
                    t.seller,
                    t.timestamp,
                ])
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        # (Keep original implementation)
        conversion_observations = {}
        for product, obs in observations.conversionObservations.items():
             # Adjust keys based on actual datamodel for the round
            conversion_observations[product] = [
                getattr(obs, 'bidPrice', None),
                getattr(obs, 'askPrice', None),
                getattr(obs, 'transportFees', None),
                getattr(obs, 'exportTariff', None), # Corrected line
                getattr(obs, 'importTariff', None),
                getattr(obs, 'sunlight', None), # Example, use actual keys
                getattr(obs, 'humidity', None)  # Example, use actual keys
            ]
        return [getattr(observations, 'plainValueObservations', {}), conversion_observations]


    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        # (Keep original implementation)
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        # (Keep original implementation)
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        # (Keep original implementation)
        if len(value) <= max_length:
            return value
        return value[: max_length - 3] + "..."

logger = Logger() # Instantiate the logger


###############################################################################
# Trader class with Simple KELP logic
###############################################################################
class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:

        # Load previous state (might not be needed for KELP anymore, but RESIN might use it indirectly or good practice)
        prev = {}
        if state.traderData:
            try:
                prev = json.loads(state.traderData)
            except Exception as e:
                logger.print(f"Error loading traderData: {e}")
                prev = {}

        res = {} # Final orders dictionary
        POSITION_LIMITS = {'KELP': 50} # Keep only KELP limit

        for product in state.order_depths:
            depth = state.order_depths[product]
            orders: List[Order] = [] # Orders for THIS product only
            pos = state.position.get(product, 0)
            pos_limit = POSITION_LIMITS.get(product, 50)
            buy_capacity = pos_limit - pos
            sell_capacity = pos + pos_limit

            # --- Common calculations (Best bid/ask needed for simple KELP buy) ---
            best_bid = 0
            best_ask = float('inf')
            best_bid_volume = 0
            best_ask_volume = 0

            if depth.buy_orders:
                best_bid = max(depth.buy_orders.keys())
                best_bid_volume = depth.buy_orders[best_bid]
            if depth.sell_orders:
                best_ask = min(depth.sell_orders.keys())
                best_ask_volume = depth.sell_orders[best_ask] # Sell volume is negative




            # =============== KELP logic (Buy 1 and Hold) ===============
            if product == "KELP":
                logger.print(f"KELP: Current position: {pos}")

                # --- Buy Logic ---
                # Check if we have less than 1 KELP
                if pos < 1:
                    # Check if there are any sell orders in the market to buy from
                    if depth.sell_orders:
                        # Find the cheapest price someone is selling at
                        best_ask_price = min(depth.sell_orders.keys()) # Same as best_ask calculated above

                        # Calculate the quantity needed to reach a position of 1
                        quantity_to_buy = 1 - pos
                        
                        # Ensure we have capacity (should always be true for qty 1 if limit > 1)
                        if quantity_to_buy > 0 and buy_capacity >= quantity_to_buy :
                             logger.print(f"KELP: Attempting to buy {quantity_to_buy} unit(s) at {best_ask_price} to reach position 1")
                             # Place the order to buy at the best available price
                             orders.append(Order(product, 2027, quantity_to_buy))
                        # else: # Optional logging if buy is blocked by capacity
                        #    logger.print(f"KELP: Want to buy {quantity_to_buy}, but buy_capacity ({buy_capacity}) is insufficient.")

                    else:
                        # No sellers in the market right now
                        logger.print("KELP: Want to buy to reach position 1, but no sell orders available.")
                
                # --- Hold Logic (No Sell Orders) ---
                # If pos is >= 1, we do nothing, effectively holding.
                # No code needed here to explicitly "hold". Just don't place sell orders.
                elif pos >= 1:
                     logger.print(f"KELP: Already holding {pos} unit(s), not placing any orders.")


            # --- Store orders for the current product ---
            res[product] = orders

        # Save state and return results
        # Note: `prev` (history) is saved but no longer actively used by the new KELP logic
        traderData = json.dumps(prev)
        conversions = 0

        logger.flush(state, res, conversions, traderData)
        return res, conversions, traderData