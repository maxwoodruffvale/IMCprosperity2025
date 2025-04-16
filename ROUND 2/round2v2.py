# Full Trader Code with HIGHER Z-Score Tiers for Basket Execution

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List, Dict
import numpy as np
import json
import math

# Logger Class (keep as provided before)
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750 # Adjust if needed

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        # Simple append, truncation happens in flush
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        # Compressing state components individually
        compressed_state = self.compress_state(state, "") # Get structure without traderData
        compressed_orders = self.compress_orders(orders)
        compressed_conv = conversions
        log_part = self.logs

        base_json = self.to_json([compressed_state, compressed_orders, compressed_conv, "", ""])
        base_length = len(base_json)
        available_length = self.max_log_length - base_length
        max_trader_data_len = available_length // 2
        max_log_len = available_length - max_trader_data_len

        truncated_state_trader_data = self.truncate(state.traderData, max_trader_data_len)
        compressed_state[1] = truncated_state_trader_data # Update traderData

        truncated_trader_data = self.truncate(trader_data, max_trader_data_len)
        truncated_logs = self.truncate(self.logs, max_log_len)

        final_output = self.to_json([compressed_state, compressed_orders, conversions, truncated_trader_data, truncated_logs])

        # Ensure final output respects the limit absolutely
        if len(final_output) > self.max_log_length:
             extra_truncation = len(final_output) - self.max_log_length + 3
             if extra_truncation < len(truncated_logs): truncated_logs = truncated_logs[:-extra_truncation] + "..."
             else:
                  truncated_logs = ""
                  extra_truncation = len(final_output) - self.max_log_length + 3
                  if extra_truncation < len(truncated_trader_data): truncated_trader_data = truncated_trader_data[:-extra_truncation] + "..."
                  else: truncated_trader_data = "TRUNC_ERR"
             final_output = self.to_json([compressed_state, compressed_orders, conversions, truncated_trader_data, truncated_logs])

        print(final_output)
        self.logs = "" # Clear logs after flushing

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        observations_compressed = None
        if state.observations: observations_compressed = self.compress_observations(state.observations)
        return [state.timestamp, trader_data, self.compress_listings(state.listings),
                self.compress_order_depths(state.order_depths), self.compress_trades(state.own_trades),
                self.compress_trades(state.market_trades), state.position, observations_compressed]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values(): compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            buy_orders_items = list(order_depth.buy_orders.items()) if isinstance(order_depth.buy_orders, dict) else []
            sell_orders_items = list(order_depth.sell_orders.items()) if isinstance(order_depth.sell_orders, dict) else []
            compressed[symbol] = [buy_orders_items, sell_orders_items]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        if trades:
            for arr in trades.values():
                for trade in arr: compressed.append([trade.symbol, trade.price, trade.quantity, trade.buyer, trade.seller, trade.timestamp])
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        plain_obs = observations.plainValueObservations if hasattr(observations, 'plainValueObservations') else {}
        conv_obs_comp = {}
        if hasattr(observations, 'conversionObservations') and observations.conversionObservations:
            for product, observation in observations.conversionObservations.items():
                if observation:
                    conv_obs_comp[product] = [
                        getattr(observation, 'bidPrice', None), getattr(observation, 'askPrice', None),
                        getattr(observation, 'transportFees', None), getattr(observation, 'exportTariff', None),
                        getattr(observation, 'importTariff', None), getattr(observation, 'sunlight', None),
                        getattr(observation, 'humidity', None)
                     ]
        return [plain_obs, conv_obs_comp]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr: compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if not isinstance(value, str): value = str(value)
        if len(value) <= max_length: return value
        return value[: max_length - 3] + "..."

logger = Logger()

class Trader:
    POSITION_LIMITS = {
        "RAINFOREST_RESIN": 50, "KELP": 50, "SQUID_INK": 50,
        "CROISSANTS": 250, "JAMS": 350, "DJEMBES": 60,
        "PICNIC_BASKET1": 60, "PICNIC_BASKET2": 100
    }
    epsilon = 1e-10

    def get_position(self, state: TradingState, product: str) -> int:
         return state.position.get(product, 0)

    def get_best_bid_ask(self, order_depth: OrderDepth) -> tuple[int | None, int | None]:
         best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
         best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
         return best_bid, best_ask

    def get_volume_at_price(self, orders: Dict[int, int], price: int | None) -> int:
         if price is None or not orders: return 0
         return abs(orders.get(price, 0))

    # --- Single Product Strategies (RESIN, KELP, INK - keep as before) ---
    # [Paste the working versions of resin_strategy, kelp_strategy, ink_strategy here]
    # Resin Strategy
    def resin_strategy(self, state: TradingState, limit: int) -> List[Order]:
        orders: List[Order] = []; product = "RAINFOREST_RESIN"
        if product not in state.order_depths: return orders
        order_depth = state.order_depths[product]; position = self.get_position(state, product)
        fair_value = 10000.0; take_width = 1.0; mm_bid_edge = 8.0; mm_ask_edge = 8.0
        buy_order_volume = 0; sell_order_volume = 0
        best_bid, best_ask = self.get_best_bid_ask(order_depth)
        if best_ask is not None and best_ask <= fair_value - take_width:
            vol = self.get_volume_at_price(order_depth.sell_orders, best_ask); qty = min(vol, limit - position)
            if qty > 0: orders.append(Order(product, best_ask, qty)); buy_order_volume += qty; #logger.print(f"TAKE {product}: Buy {qty} @ {best_ask}")
        if best_bid is not None and best_bid >= fair_value + take_width:
            vol = self.get_volume_at_price(order_depth.buy_orders, best_bid); qty = min(vol, limit + position)
            if qty > 0: orders.append(Order(product, best_bid, -qty)); sell_order_volume += qty; #logger.print(f"TAKE {product}: Sell {qty} @ {best_bid}")
        position_after_take = position + buy_order_volume - sell_order_volume; bid_price = round(fair_value - mm_bid_edge); ask_price = round(fair_value + mm_ask_edge)
        qty_to_buy = limit - position_after_take
        if qty_to_buy > 0: orders.append(Order(product, bid_price, qty_to_buy)); #logger.print(f"MM {product}: Bid {qty_to_buy} @ {bid_price}")
        qty_to_sell = limit + position_after_take
        if qty_to_sell > 0: orders.append(Order(product, ask_price, -qty_to_sell)); #logger.print(f"MM {product}: Ask {qty_to_sell} @ {ask_price}")
        return orders

    # Kelp Strategy
    def kelp_strategy(self, state: TradingState, limit: int) -> List[Order]:
        orders: List[Order] = []; product = "KELP"
        if product not in state.order_depths: return orders
        order_depth = state.order_depths[product]; position = self.get_position(state, product)
        best_bid, best_ask = self.get_best_bid_ask(order_depth)
        if best_bid is None or best_ask is None: return orders
        fair_value = (best_ask + best_bid) / 2.0; take_width = 1.0; mm_spread = 2.0
        buy_order_volume = 0; sell_order_volume = 0
        if best_ask <= fair_value - take_width:
            vol = self.get_volume_at_price(order_depth.sell_orders, best_ask); qty = min(vol, limit - position)
            if qty > 0: orders.append(Order(product, best_ask, qty)); buy_order_volume += qty; #logger.print(f"TAKE {product}: Buy {qty} @ {best_ask}")
        if best_bid >= fair_value + take_width:
            vol = self.get_volume_at_price(order_depth.buy_orders, best_bid); qty = min(vol, limit + position)
            if qty > 0: orders.append(Order(product, best_bid, -qty)); sell_order_volume += qty; #logger.print(f"TAKE {product}: Sell {qty} @ {best_bid}")
        position_after_take = position + buy_order_volume - sell_order_volume; bid_price = math.floor(fair_value - mm_spread / 2.0); ask_price = math.ceil(fair_value + mm_spread / 2.0)
        qty_to_buy = limit - position_after_take
        if qty_to_buy > 0: orders.append(Order(product, bid_price, qty_to_buy)); #logger.print(f"MM {product}: Bid {qty_to_buy} @ {bid_price}")
        qty_to_sell = limit + position_after_take
        if qty_to_sell > 0: orders.append(Order(product, ask_price, -qty_to_sell)); #logger.print(f"MM {product}: Ask {qty_to_sell} @ {ask_price}")
        return orders

    # Ink Strategy
    def ink_strategy(self, state: TradingState, limit: int, stored_data):
        orders: List[Order] = []; product = "SQUID_INK"
        if product not in state.order_depths: return orders
        order_depth = state.order_depths[product]; position = self.get_position(state, product)
        best_bid, best_ask = self.get_best_bid_ask(order_depth)
        if best_bid is None or best_ask is None: return orders
        current_price = (best_ask + best_bid) / 2.0
        window = 50; z_score_threshold = 2.0
        history = stored_data[product].setdefault("price_history", [])
        history.append(current_price)
        if len(history) > window: history.pop(0)
        buy_order_volume = 0; sell_order_volume = 0
        if len(history) >= window:
            mean_price = np.mean(history); std_price = np.std(history)
            if std_price > 1e-6:
                z_score = (current_price - mean_price) / std_price
                # logger.print(f"INK Z-Score: {z_score:.2f}") # Optional
                if z_score > z_score_threshold:
                    if best_bid is not None:
                         vol = self.get_volume_at_price(order_depth.buy_orders, best_bid); qty = min(vol, limit + position)
                         if qty > 0: orders.append(Order(product, best_bid, -qty)); sell_order_volume += qty; #logger.print(f"MR {product}: Sell {qty} @ {best_bid}")
                elif z_score < -z_score_threshold:
                    if best_ask is not None:
                        vol = self.get_volume_at_price(order_depth.sell_orders, best_ask); qty = min(vol, limit - position)
                        if qty > 0: orders.append(Order(product, best_ask, qty)); buy_order_volume += qty; #logger.print(f"MR {product}: Buy {qty} @ {best_ask}")
        position_after_trade = position + buy_order_volume - sell_order_volume; mm_spread = 2.0
        bid_price = math.floor(current_price - mm_spread / 2.0); ask_price = math.ceil(current_price + mm_spread / 2.0)
        qty_to_buy = limit - position_after_trade
        if qty_to_buy > 0: orders.append(Order(product, bid_price, qty_to_buy)); #logger.print(f"MM {product}: Bid {qty_to_buy} @ {bid_price}")
        qty_to_sell = limit + position_after_trade
        if qty_to_sell > 0: orders.append(Order(product, ask_price, -qty_to_sell)); #logger.print(f"MM {product}: Ask {qty_to_sell} @ {ask_price}")
        return orders

    # --- Basket Calculation and Conversion Methods ---
    # [Paste the working versions of synthetic1/2/3_order_depth and convert_synthetic1/2/3_orders here]
    # Synthetic 1 (6 CR, 3 JM, 1 DJ)
    def synthetic1_order_depth(self, state: TradingState):
        components = ["CROISSANTS", "JAMS", "DJEMBES"]; ods = state.order_depths; pos = state.position; lim = self.POSITION_LIMITS
        if any(comp not in ods for comp in components): return OrderDepth()
        cr_bb, cr_ba = self.get_best_bid_ask(ods["CROISSANTS"]); jm_bb, jm_ba = self.get_best_bid_ask(ods["JAMS"]); dj_bb, dj_ba = self.get_best_bid_ask(ods["DJEMBES"])
        if None in [cr_bb, cr_ba, jm_bb, jm_ba, dj_bb, dj_ba]: return OrderDepth()
        synthetic_od = OrderDepth(); ask_p = cr_ba * 6 + jm_ba * 3 + dj_ba; bid_p = cr_bb * 6 + jm_bb * 3 + dj_bb
        cr_av = self.get_volume_at_price(ods["CROISSANTS"].sell_orders, cr_ba); jm_av = self.get_volume_at_price(ods["JAMS"].sell_orders, jm_ba); dj_av = self.get_volume_at_price(ods["DJEMBES"].sell_orders, dj_ba)
        max_b_cr = (lim["CROISSANTS"] - self.get_position(state, "CROISSANTS")) // 6; max_b_jm = (lim["JAMS"] - self.get_position(state, "JAMS")) // 3; max_b_dj = lim["DJEMBES"] - self.get_position(state, "DJEMBES")
        syn_ask_v = min(cr_av // 6, jm_av // 3, dj_av, max_b_cr, max_b_jm, max_b_dj)
        if syn_ask_v > 0: synthetic_od.sell_orders[ask_p] = -syn_ask_v
        cr_bv = self.get_volume_at_price(ods["CROISSANTS"].buy_orders, cr_bb); jm_bv = self.get_volume_at_price(ods["JAMS"].buy_orders, jm_bb); dj_bv = self.get_volume_at_price(ods["DJEMBES"].buy_orders, dj_bb)
        max_s_cr = (lim["CROISSANTS"] + self.get_position(state, "CROISSANTS")) // 6; max_s_jm = (lim["JAMS"] + self.get_position(state, "JAMS")) // 3; max_s_dj = lim["DJEMBES"] + self.get_position(state, "DJEMBES")
        syn_bid_v = min(cr_bv // 6, jm_bv // 3, dj_bv, max_s_cr, max_s_jm, max_s_dj)
        if syn_bid_v > 0: synthetic_od.buy_orders[bid_p] = syn_bid_v
        return synthetic_od

    def convert_synthetic1_orders(self, state: TradingState, synthetic_order: Order):
        comp_orders = []; qty = synthetic_order.quantity; ods = state.order_depths;
        if qty == 0: return comp_orders
        if qty > 0: cr_p, jm_p, dj_p = (min(ods[p].sell_orders.keys()) if ods[p].sell_orders else None for p in ["CROISSANTS", "JAMS", "DJEMBES"])
        else: cr_p, jm_p, dj_p = (max(ods[p].buy_orders.keys()) if ods[p].buy_orders else None for p in ["CROISSANTS", "JAMS", "DJEMBES"])
        if None in [cr_p, jm_p, dj_p]: logger.print("WARN: Missing comp price syn1 conv"); return []
        comp_orders.append(Order("CROISSANTS", cr_p, qty * 6)); comp_orders.append(Order("JAMS", jm_p, qty * 3)); comp_orders.append(Order("DJEMBES", dj_p, qty * 1))
        return comp_orders

    # Synthetic 2 (4 CR, 2 JM)
    def synthetic2_order_depth(self, state: TradingState):
        components = ["CROISSANTS", "JAMS"]; ods = state.order_depths; pos = state.position; lim = self.POSITION_LIMITS
        if any(comp not in ods for comp in components): return OrderDepth()
        cr_bb, cr_ba = self.get_best_bid_ask(ods["CROISSANTS"]); jm_bb, jm_ba = self.get_best_bid_ask(ods["JAMS"])
        if None in [cr_bb, cr_ba, jm_bb, jm_ba]: return OrderDepth()
        synthetic_od = OrderDepth(); ask_p = cr_ba * 4 + jm_ba * 2; bid_p = cr_bb * 4 + jm_bb * 2
        cr_av = self.get_volume_at_price(ods["CROISSANTS"].sell_orders, cr_ba); jm_av = self.get_volume_at_price(ods["JAMS"].sell_orders, jm_ba)
        max_b_cr = (lim["CROISSANTS"] - self.get_position(state, "CROISSANTS")) // 4; max_b_jm = (lim["JAMS"] - self.get_position(state, "JAMS")) // 2
        syn_ask_v = min(cr_av // 4, jm_av // 2, max_b_cr, max_b_jm)
        if syn_ask_v > 0: synthetic_od.sell_orders[ask_p] = -syn_ask_v
        cr_bv = self.get_volume_at_price(ods["CROISSANTS"].buy_orders, cr_bb); jm_bv = self.get_volume_at_price(ods["JAMS"].buy_orders, jm_bb)
        max_s_cr = (lim["CROISSANTS"] + self.get_position(state, "CROISSANTS")) // 4; max_s_jm = (lim["JAMS"] + self.get_position(state, "JAMS")) // 2
        syn_bid_v = min(cr_bv // 4, jm_bv // 2, max_s_cr, max_s_jm)
        if syn_bid_v > 0: synthetic_od.buy_orders[bid_p] = syn_bid_v
        return synthetic_od

    def convert_synthetic2_orders(self, state: TradingState, synthetic_order: Order):
        comp_orders = []; qty = synthetic_order.quantity; ods = state.order_depths
        if qty == 0: return comp_orders
        if qty > 0: cr_p, jm_p = (min(ods[p].sell_orders.keys()) if ods[p].sell_orders else None for p in ["CROISSANTS", "JAMS"])
        else: cr_p, jm_p = (max(ods[p].buy_orders.keys()) if ods[p].buy_orders else None for p in ["CROISSANTS", "JAMS"])
        if None in [cr_p, jm_p]: logger.print("WARN: Missing comp price syn2 conv"); return []
        comp_orders.append(Order("CROISSANTS", cr_p, qty * 4)); comp_orders.append(Order("JAMS", jm_p, qty * 2))
        return comp_orders

    # Synthetic 3 (PB1 - 1.5 * PB2 = DJ)
    def synthetic3_order_depth(self, state: TradingState):
        components = ["PICNIC_BASKET1", "PICNIC_BASKET2"]; ods = state.order_depths; pos = state.position; lim = self.POSITION_LIMITS
        if any(comp not in ods for comp in components): return OrderDepth()
        pb1_bb, pb1_ba = self.get_best_bid_ask(ods["PICNIC_BASKET1"]); pb2_bb, pb2_ba = self.get_best_bid_ask(ods["PICNIC_BASKET2"])
        if None in [pb1_bb, pb1_ba, pb2_bb, pb2_ba]: return OrderDepth()
        synthetic_od = OrderDepth(); pb2_ratio = 1.5; pb1_ratio = 1 # Ratio for PB1 is 1
        ask_p = pb1_ba - int(round(pb2_ratio * pb2_bb)); bid_p = pb1_bb - int(round(pb2_ratio * pb2_ba))
        pb1_av = self.get_volume_at_price(ods["PICNIC_BASKET1"].sell_orders, pb1_ba); pb2_bv = self.get_volume_at_price(ods["PICNIC_BASKET2"].buy_orders, pb2_bb)
        max_b_pb1 = lim["PICNIC_BASKET1"] - self.get_position(state, "PICNIC_BASKET1"); max_s_pb2 = lim["PICNIC_BASKET2"] + self.get_position(state, "PICNIC_BASKET2")
        syn_ask_v = min(pb1_av // pb1_ratio, int(pb2_bv // pb2_ratio), max_b_pb1 // pb1_ratio, int(max_s_pb2 // pb2_ratio))
        if syn_ask_v > 0: synthetic_od.sell_orders[ask_p] = -syn_ask_v
        pb1_bv = self.get_volume_at_price(ods["PICNIC_BASKET1"].buy_orders, pb1_bb); pb2_av = self.get_volume_at_price(ods["PICNIC_BASKET2"].sell_orders, pb2_ba)
        max_s_pb1 = lim["PICNIC_BASKET1"] + self.get_position(state, "PICNIC_BASKET1"); max_b_pb2 = lim["PICNIC_BASKET2"] - self.get_position(state, "PICNIC_BASKET2")
        syn_bid_v = min(pb1_bv // pb1_ratio, int(pb2_av // pb2_ratio), max_s_pb1 // pb1_ratio, int(max_b_pb2 // pb2_ratio))
        if syn_bid_v > 0: synthetic_od.buy_orders[bid_p] = syn_bid_v
        return synthetic_od

    def convert_synthetic3_orders(self, state: TradingState, synthetic_order: Order):
        comp_orders = []; qty = synthetic_order.quantity; ods = state.order_depths
        if qty == 0: return comp_orders
        pb1_qty = qty * 1; pb2_qty = -int(round(qty * 1.5))
        if pb1_qty > 0: pb1_p = min(ods["PICNIC_BASKET1"].sell_orders.keys()) if ods["PICNIC_BASKET1"].sell_orders else None
        elif pb1_qty < 0: pb1_p = max(ods["PICNIC_BASKET1"].buy_orders.keys()) if ods["PICNIC_BASKET1"].buy_orders else None
        else: pb1_p = None
        if pb2_qty > 0: pb2_p = min(ods["PICNIC_BASKET2"].sell_orders.keys()) if ods["PICNIC_BASKET2"].sell_orders else None
        elif pb2_qty < 0: pb2_p = max(ods["PICNIC_BASKET2"].buy_orders.keys()) if ods["PICNIC_BASKET2"].buy_orders else None
        else: pb2_p = None
        if None in [pb1_p, pb2_p]: logger.print("WARN: Missing comp price syn3 conv"); return []
        if pb1_qty != 0: comp_orders.append(Order("PICNIC_BASKET1", pb1_p, pb1_qty))
        if pb2_qty != 0: comp_orders.append(Order("PICNIC_BASKET2", pb2_p, pb2_qty))
        return comp_orders

    # --- Basket Arbitrage Strategies with Tiered Execution ---

    def picnic1_strategy(self, state: TradingState, limit: int, stored_data):
        # --- Tiered Z-Score Arb for PICNIC_BASKET1 = 6 CR + 3 JM + 1 DJ ---
        orders: List[Order] = []; croissants_orders: List[Order] = []; jams_orders: List[Order] = []; djembes_orders: List[Order] = []
        product = "PICNIC_BASKET1"; components = ["CROISSANTS", "JAMS", "DJEMBES"]
        if product not in state.order_depths or any(comp not in state.order_depths for comp in components): return orders, croissants_orders, jams_orders, djembes_orders

        order_depth = state.order_depths[product]; position = self.get_position(state, product)
        best_bid, best_ask = self.get_best_bid_ask(order_depth)
        if best_bid is None or best_ask is None: return orders, croissants_orders, jams_orders, djembes_orders
        picnic_mid_price = (best_ask + best_bid) / 2.0

        try: # Calculate synthetic price
            synthetic_od = self.synthetic1_order_depth(state); synth_bb, synth_ba = self.get_best_bid_ask(synthetic_od)
            if synth_bb is None or synth_ba is None: return orders, croissants_orders, jams_orders, djembes_orders
            synthetic_mid_price = (synth_ba + synth_bb) / 2.0
        except Exception as e: logger.print(f"ERROR: Calc synth1 failed: {e}"); return orders, croissants_orders, jams_orders, djembes_orders

        # Calculate Z-Score
        spread = picnic_mid_price - synthetic_mid_price; hardcoded_mean = 48.745; window = 30
        history = stored_data[product].setdefault("spread_history", []); history.append(spread)
        if len(history) > window: history.pop(0)
        if len(history) < 5: return orders, croissants_orders, jams_orders, djembes_orders
        spread_std = np.std(history);
        if spread_std < 1e-6: return orders, croissants_orders, jams_orders, djembes_orders
        z_score = (spread - hardcoded_mean) / spread_std
        # logger.print(f"PICNIC1 Z: {z_score:.2f}")

        # Tiered execution parameters
        z_levels = [20.0, 25.0, 30.0]  # <<-- HIGHER Z-SCORE THRESHOLDS
        position_scales = [0.33, 0.66, 1.0] # Example: Target 1/3, 2/3, full limit

        desired_position = 0; trade_direction = 0; abs_z = abs(z_score); target_scale = 0.0
        if abs_z >= z_levels[0]:
            if abs_z >= z_levels[2]: target_scale = position_scales[2]
            elif abs_z >= z_levels[1]: target_scale = position_scales[1]
            else: target_scale = position_scales[0]
            if z_score > 0: desired_position = -int(round(limit * target_scale)); trade_direction = -1 if desired_position < position else 0
            else: desired_position = int(round(limit * target_scale)); trade_direction = 1 if desired_position > position else 0

        # Calculate and execute trade
        execute_volume = 0
        if trade_direction == -1: # Sell Picnic, Buy Synthetic
            target_qty = position - desired_position; picnic_price = best_bid
            picnic_vol = self.get_volume_at_price(order_depth.buy_orders, picnic_price); synth_price = synth_ba
            synth_vol = self.get_volume_at_price(synthetic_od.sell_orders, synth_price)
            execute_volume = min(target_qty, picnic_vol, synth_vol)
            if execute_volume > 0:
                 orders.append(Order(product, picnic_price, -execute_volume))
                 component_orders = self.convert_synthetic1_orders(state, Order("SYNTHETIC1", synth_price, execute_volume))
                 croissants_orders.extend([o for o in component_orders if o.symbol=="CROISSANTS"])
                 jams_orders.extend([o for o in component_orders if o.symbol=="JAMS"])
                 djembes_orders.extend([o for o in component_orders if o.symbol=="DJEMBES"])
        elif trade_direction == 1: # Buy Picnic, Sell Synthetic
            target_qty = desired_position - position; picnic_price = best_ask
            picnic_vol = self.get_volume_at_price(order_depth.sell_orders, picnic_price); synth_price = synth_bb
            synth_vol = self.get_volume_at_price(synthetic_od.buy_orders, synth_price)
            execute_volume = min(target_qty, picnic_vol, synth_vol)
            if execute_volume > 0:
                orders.append(Order(product, picnic_price, execute_volume))
                component_orders = self.convert_synthetic1_orders(state, Order("SYNTHETIC1", synth_price, -execute_volume))
                croissants_orders.extend([o for o in component_orders if o.symbol=="CROISSANTS"])
                jams_orders.extend([o for o in component_orders if o.symbol=="JAMS"])
                djembes_orders.extend([o for o in component_orders if o.symbol=="DJEMBES"])

        # logger.print(f"PICNIC1 Strat: Z={z_score:.2f}, CurrPos={position}, DesPos={desired_position}, ExecVol={execute_volume if trade_direction!=0 else 0}")
        return orders, croissants_orders, jams_orders, djembes_orders

    def picnic2_strategy(self, state: TradingState, limit: int, stored_data):
        # --- Tiered Z-Score Arb for PICNIC_BASKET2 = 4 CR + 2 JM ---
        orders: List[Order] = []; croissants_orders: List[Order] = []; jams_orders: List[Order] = []
        product = "PICNIC_BASKET2"; components = ["CROISSANTS", "JAMS"]
        if product not in state.order_depths or any(comp not in state.order_depths for comp in components): return orders, croissants_orders, jams_orders

        order_depth = state.order_depths[product]; position = self.get_position(state, product)
        best_bid, best_ask = self.get_best_bid_ask(order_depth)
        if best_bid is None or best_ask is None: return orders, croissants_orders, jams_orders
        picnic_mid_price = (best_ask + best_bid) / 2.0

        try: # Calculate synthetic price
            synthetic_od = self.synthetic2_order_depth(state); synth_bb, synth_ba = self.get_best_bid_ask(synthetic_od)
            if synth_bb is None or synth_ba is None: return orders, croissants_orders, jams_orders
            synthetic_mid_price = (synth_ba + synth_bb) / 2.0
        except Exception as e: logger.print(f"ERROR: Calc synth2 failed: {e}"); return orders, croissants_orders, jams_orders

        # Calculate Z-Score
        spread = picnic_mid_price - synthetic_mid_price; hardcoded_mean = 30.237; window = 20 # Shorter window maybe?
        history = stored_data[product].setdefault("spread_history", []); history.append(spread)
        if len(history) > window: history.pop(0)
        if len(history) < 5: return orders, croissants_orders, jams_orders
        spread_std = np.std(history);
        if spread_std < 1e-6: return orders, croissants_orders, jams_orders
        z_score = (spread - hardcoded_mean) / spread_std
        # logger.print(f"PICNIC2 Z: {z_score:.2f}")

        # Tiered execution parameters (Higher thresholds based on original 60)
        z_levels = [30.0, 45.0, 60.0] # <<-- HIGHER Z-SCORE THRESHOLDS
        position_scales = [0.33, 0.66, 1.0]

        desired_position = 0; trade_direction = 0; abs_z = abs(z_score); target_scale = 0.0
        if abs_z >= z_levels[0]:
            if abs_z >= z_levels[2]: target_scale = position_scales[2]
            elif abs_z >= z_levels[1]: target_scale = position_scales[1]
            else: target_scale = position_scales[0]
            if z_score > 0: desired_position = -int(round(limit * target_scale)); trade_direction = -1 if desired_position < position else 0
            else: desired_position = int(round(limit * target_scale)); trade_direction = 1 if desired_position > position else 0

        # Calculate and execute trade
        execute_volume = 0
        if trade_direction == -1: # Sell Picnic, Buy Synthetic
            target_qty = position - desired_position; picnic_price = best_bid
            picnic_vol = self.get_volume_at_price(order_depth.buy_orders, picnic_price); synth_price = synth_ba
            synth_vol = self.get_volume_at_price(synthetic_od.sell_orders, synth_price)
            execute_volume = min(target_qty, picnic_vol, synth_vol)
            if execute_volume > 0:
                 orders.append(Order(product, picnic_price, -execute_volume))
                 component_orders = self.convert_synthetic2_orders(state, Order("SYNTHETIC2", synth_price, execute_volume))
                 croissants_orders.extend([o for o in component_orders if o.symbol=="CROISSANTS"])
                 jams_orders.extend([o for o in component_orders if o.symbol=="JAMS"])
        elif trade_direction == 1: # Buy Picnic, Sell Synthetic
            target_qty = desired_position - position; picnic_price = best_ask
            picnic_vol = self.get_volume_at_price(order_depth.sell_orders, picnic_price); synth_price = synth_bb
            synth_vol = self.get_volume_at_price(synthetic_od.buy_orders, synth_price)
            execute_volume = min(target_qty, picnic_vol, synth_vol)
            if execute_volume > 0:
                orders.append(Order(product, picnic_price, execute_volume))
                component_orders = self.convert_synthetic2_orders(state, Order("SYNTHETIC2", synth_price, -execute_volume))
                croissants_orders.extend([o for o in component_orders if o.symbol=="CROISSANTS"])
                jams_orders.extend([o for o in component_orders if o.symbol=="JAMS"])

        # logger.print(f"PICNIC2 Strat: Z={z_score:.2f}, CurrPos={position}, DesPos={desired_position}, ExecVol={execute_volume if trade_direction!=0 else 0}")
        return orders, croissants_orders, jams_orders

    def djembes_strategy(self, state: TradingState, limit: int, stored_data):
         # --- Tiered Z-Score Arb for DJEMBES = PB1 - 1.5 * PB2 ---
        orders: List[Order] = []; picnic1_orders: List[Order] = []; picnic2_orders: List[Order] = []
        product = "DJEMBES"; components = ["PICNIC_BASKET1", "PICNIC_BASKET2"]
        if product not in state.order_depths or any(comp not in state.order_depths for comp in components): return orders, picnic1_orders, picnic2_orders

        order_depth = state.order_depths[product]; position = self.get_position(state, product)
        best_bid, best_ask = self.get_best_bid_ask(order_depth)
        if best_bid is None or best_ask is None: return orders, picnic1_orders, picnic2_orders
        djembes_mid_price = (best_ask + best_bid) / 2.0

        try: # Calculate synthetic price
            synthetic_od = self.synthetic3_order_depth(state); synth_bb, synth_ba = self.get_best_bid_ask(synthetic_od)
            if synth_bb is None or synth_ba is None: return orders, picnic1_orders, picnic2_orders
            synthetic_mid_price = (synth_ba + synth_bb) / 2.0
        except Exception as e: logger.print(f"ERROR: Calc synth3 failed: {e}"); return orders, picnic1_orders, picnic2_orders

        # Calculate Z-Score
        spread = djembes_mid_price - synthetic_mid_price; hardcoded_mean = -3.389; window = 30
        history = stored_data[product].setdefault("spread_history", []); history.append(spread)
        if len(history) > window: history.pop(0)
        if len(history) < 5: return orders, picnic1_orders, picnic2_orders
        spread_std = np.std(history);
        if spread_std < 1e-6: return orders, picnic1_orders, picnic2_orders
        z_score = (spread - hardcoded_mean) / spread_std
        # logger.print(f"DJEMBES Z: {z_score:.2f}")

        # Tiered execution parameters
        z_levels = [20.0, 25.0, 30.0] # <<-- HIGHER Z-SCORE THRESHOLDS
        position_scales = [0.33, 0.66, 1.0]

        desired_position = 0; trade_direction = 0; abs_z = abs(z_score); target_scale = 0.0
        if abs_z >= z_levels[0]:
            if abs_z >= z_levels[2]: target_scale = position_scales[2]
            elif abs_z >= z_levels[1]: target_scale = position_scales[1]
            else: target_scale = position_scales[0]
            # Note: Positive Z means DJ is expensive relative to synthetic
            if z_score > 0: desired_position = -int(round(limit * target_scale)); trade_direction = -1 if desired_position < position else 0
            else: desired_position = int(round(limit * target_scale)); trade_direction = 1 if desired_position > position else 0

        # Calculate and execute trade
        execute_volume = 0
        if trade_direction == -1: # Sell Djembe, Buy Synthetic3 (Buy PB1, Sell PB2)
            target_qty = position - desired_position; djembe_price = best_bid
            djembe_vol = self.get_volume_at_price(order_depth.buy_orders, djembe_price); synth_price = synth_ba # Use synth ask
            synth_vol = self.get_volume_at_price(synthetic_od.sell_orders, synth_price)
            execute_volume = min(target_qty, djembe_vol, synth_vol)
            if execute_volume > 0:
                 orders.append(Order(product, djembe_price, -execute_volume))
                 component_orders = self.convert_synthetic3_orders(state, Order("SYNTHETIC3", synth_price, execute_volume))
                 picnic1_orders.extend([o for o in component_orders if o.symbol=="PICNIC_BASKET1"])
                 picnic2_orders.extend([o for o in component_orders if o.symbol=="PICNIC_BASKET2"])
        elif trade_direction == 1: # Buy Djembe, Sell Synthetic3 (Sell PB1, Buy PB2)
            target_qty = desired_position - position; djembe_price = best_ask
            djembe_vol = self.get_volume_at_price(order_depth.sell_orders, djembe_price); synth_price = synth_bb # Use synth bid
            synth_vol = self.get_volume_at_price(synthetic_od.buy_orders, synth_price)
            execute_volume = min(target_qty, djembe_vol, synth_vol)
            if execute_volume > 0:
                orders.append(Order(product, djembe_price, execute_volume))
                component_orders = self.convert_synthetic3_orders(state, Order("SYNTHETIC3", synth_price, -execute_volume))
                picnic1_orders.extend([o for o in component_orders if o.symbol=="PICNIC_BASKET1"])
                picnic2_orders.extend([o for o in component_orders if o.symbol=="PICNIC_BASKET2"])

        # logger.print(f"DJEMBES Strat: Z={z_score:.2f}, CurrPos={position}, DesPos={desired_position}, ExecVol={execute_volume if trade_direction!=0 else 0}")
        return orders, picnic1_orders, picnic2_orders

    # --- Spread Picker (remains the same) ---
    def spread_picker(self, state: TradingState, arb1_possible: bool, arb2_possible: bool, arb3_possible: bool):
        spread1, spread2, spread3 = 0, 0, 0
        # Calculate Spread 1
        if arb1_possible:
             try: # Wrap in try-except
                 pb1_od=state.order_depths["PICNIC_BASKET1"]; pb1_bb,pb1_ba=self.get_best_bid_ask(pb1_od)
                 syn1_od=self.synthetic1_order_depth(state); syn1_bb,syn1_ba=self.get_best_bid_ask(syn1_od)
                 if None not in [pb1_bb, pb1_ba, syn1_bb, syn1_ba]: spread1 = ((pb1_ba + pb1_bb)/2.0) - ((syn1_ba + syn1_bb)/2.0) - 48.745
             except: pass
        # Calculate Spread 2
        if arb2_possible:
            try:
                pb2_od=state.order_depths["PICNIC_BASKET2"]; pb2_bb,pb2_ba=self.get_best_bid_ask(pb2_od)
                syn2_od=self.synthetic2_order_depth(state); syn2_bb,syn2_ba=self.get_best_bid_ask(syn2_od)
                if None not in [pb2_bb, pb2_ba, syn2_bb, syn2_ba]: spread2 = ((pb2_ba + pb2_bb)/2.0) - ((syn2_ba + syn2_bb)/2.0) - 30.237
            except: pass
        # Calculate Spread 3
        if arb3_possible:
            try:
                dj_od=state.order_depths["DJEMBES"]; dj_bb,dj_ba=self.get_best_bid_ask(dj_od)
                syn3_od=self.synthetic3_order_depth(state); syn3_bb,syn3_ba=self.get_best_bid_ask(syn3_od)
                if None not in [dj_bb, dj_ba, syn3_bb, syn3_ba]: spread3 = ((dj_ba + dj_bb)/2.0) - ((syn3_ba + syn3_bb)/2.0) - (-3.389)
            except: pass
        # Find best
        abs_spreads = {"PICNIC_BASKET1": abs(spread1), "PICNIC_BASKET2": abs(spread2), "DJEMBES": abs(spread3)}
        non_zero_spreads = {k: v for k, v in abs_spreads.items() if v > 1e-6}
        if not non_zero_spreads: return None
        best = max(non_zero_spreads, key=non_zero_spreads.get)
        # Check possibility again
        if best == "PICNIC_BASKET1" and not arb1_possible: return None
        if best == "PICNIC_BASKET2" and not arb2_possible: return None
        if best == "DJEMBES" and not arb3_possible: return None
        return best

    # --- Main Run Method (Adjusted Aggregation) ---
    def run(self, state: TradingState):
        stored_data = json.loads(state.traderData) if state.traderData else {}
        all_products = list(self.POSITION_LIMITS.keys())
        for product in all_products:
             if product not in stored_data:
                 stored_data[product] = {"spread_history": [], "price_history": []}

        result = {} # Final orders dictionary

        # --- Run Single Product Strategies ---
        for product in ["RAINFOREST_RESIN", "KELP", "SQUID_INK"]:
             if product in state.order_depths:
                 limit = self.POSITION_LIMITS.get(product, 0)
                 strat_orders = []
                 if product == "RAINFOREST_RESIN": strat_orders = self.resin_strategy(state, limit)
                 elif product == "KELP": strat_orders = self.kelp_strategy(state, limit)
                 elif product == "SQUID_INK": strat_orders = self.ink_strategy(state, limit, stored_data)
                 result[product] = strat_orders # Assign orders directly

        # --- Run Basket Arbitrage Logic ---
        arb1_possible = all(p in state.order_depths and state.order_depths[p] for p in ["PICNIC_BASKET1", "CROISSANTS", "JAMS", "DJEMBES"])
        arb2_possible = all(p in state.order_depths and state.order_depths[p] for p in ["PICNIC_BASKET2", "CROISSANTS", "JAMS"])
        arb3_possible = all(p in state.order_depths and state.order_depths[p] for p in ["DJEMBES", "PICNIC_BASKET1", "PICNIC_BASKET2"])

        best_arb_target = self.spread_picker(state, arb1_possible, arb2_possible, arb3_possible)
        logger.print(f"Best Arb Target: {best_arb_target}")

        temp_arb_orders = {} # Store orders from the chosen arb
        if best_arb_target == "PICNIC_BASKET1" and arb1_possible:
             pb1_o, cr_o, jm_o, dj_o = self.picnic1_strategy(state, self.POSITION_LIMITS["PICNIC_BASKET1"], stored_data)
             temp_arb_orders = {"PICNIC_BASKET1":pb1_o, "CROISSANTS":cr_o, "JAMS":jm_o, "DJEMBES":dj_o}
        elif best_arb_target == "PICNIC_BASKET2" and arb2_possible:
             pb2_o, cr_o, jm_o = self.picnic2_strategy(state, self.POSITION_LIMITS["PICNIC_BASKET2"], stored_data)
             temp_arb_orders = {"PICNIC_BASKET2":pb2_o, "CROISSANTS":cr_o, "JAMS":jm_o}
        elif best_arb_target == "DJEMBES" and arb3_possible:
             dj_o, pb1_o, pb2_o = self.djembes_strategy(state, self.POSITION_LIMITS["DJEMBES"], stored_data)
             temp_arb_orders = {"DJEMBES":dj_o, "PICNIC_BASKET1":pb1_o, "PICNIC_BASKET2":pb2_o}

        # Aggregate orders: Add arb orders to any existing orders from single product strats
        for product, orders_list in temp_arb_orders.items():
             if product not in result: result[product] = [] # Initialize if product wasn't in single strats
             result[product].extend(orders_list) # Append new orders

        # Final data prep for return
        trader_data = json.dumps(stored_data, separators=(",", ":"))
        conversions = 0

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data