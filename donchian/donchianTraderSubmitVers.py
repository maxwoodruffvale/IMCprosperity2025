import jsonpickle
import numpy as np
import pandas as pd
import math
from typing import List, Dict
import statistics
from datamodel import TradingState, Order, OrderDepth, Trade

class Trader:
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        # Retrieve persistent state from traderData (if any)
        if state.traderData:
            try:
                data = jsonpickle.decode(state.traderData)
            except Exception:
                data = {}
        else:
            data = {}

        # Initialize persistent structures if not present
        if 'price_history' not in data:
            data['price_history'] = {}  # product -> list of midprices
        if 'breakout_confirmations' not in data:
            data['breakout_confirmations'] = {}  # product -> {'signal': None/'buy'/'sell', 'count': int}

        orders_result = {}

        # Iterate over all products available in order_depths
        for product, order_depth in state.order_depths.items():
            # Compute the midprice from best bid and best ask if possible
            midprice = None
            if order_depth.buy_orders and order_depth.sell_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                midprice = (best_bid + best_ask) / 2
            elif order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders.keys())
                midprice = best_bid
            elif order_depth.sell_orders:
                best_ask = min(order_depth.sell_orders.keys())
                midprice = best_ask

            if midprice is None:
                continue  # No price info available, skip this product

            # Update price history for the product
            if product not in data['price_history']:
                data['price_history'][product] = []
            data['price_history'][product].append(midprice)
            # Keep only the latest 200 prices
            if len(data['price_history'][product]) > 200:
                data['price_history'][product] = data['price_history'][product][-200:]

            price_series = data['price_history'][product]
            # Calculate volatility (standard deviation of price differences)
            if len(price_series) >= 2:
                returns = np.diff(price_series)
                volatility = np.std(returns)
            else:
                volatility = 0.0

            # Determine optimal lookback period based on volatility.
            # Here we use a simple formula: optimal_lookback = 50 / (volatility + epsilon),
            # clamped between 10 and 100 iterations.
            epsilon = 1e-5
            optimal_lookback = int(np.clip(50 / (volatility + epsilon), 10, 100))
            window = price_series[-optimal_lookback:] if len(price_series) >= optimal_lookback else price_series
            rolling_high = max(window)
            rolling_low = min(window)

            # Initialize breakout confirmation storage for this product if not present.
            if product not in data['breakout_confirmations']:
                data['breakout_confirmations'][product] = {'signal': None, 'count': 0}

            # Determine breakout signal: use a threshold of 0.5% beyond the rolling high/low.
            signal = None
            if midprice > rolling_high * 1.005:
                signal = 'buy'
            elif midprice < rolling_low * 0.995:
                signal = 'sell'

            # Update confirmation count: require two consecutive signals
            if signal == data['breakout_confirmations'][product]['signal']:
                data['breakout_confirmations'][product]['count'] += 1
            else:
                data['breakout_confirmations'][product]['signal'] = signal
                data['breakout_confirmations'][product]['count'] = 1

            confirmed = data['breakout_confirmations'][product]['count'] >= 2

            # Compute order book imbalance as additional confirmation.
            # Imbalance = (total buy volume - total sell volume) / (total buy volume + total sell volume)
            total_buy = sum(order_depth.buy_orders.values()) if order_depth.buy_orders else 0
            total_sell = abs(sum(order_depth.sell_orders.values())) if order_depth.sell_orders else 0
            imbalance = 0
            if (total_buy + total_sell) > 0:
                imbalance = (total_buy - total_sell) / (total_buy + total_sell)

            # Set liquidity threshold: require imbalance > 0.2 for buy signals and < -0.2 for sell signals.
            liquidity_confirm = False
            if signal == 'buy' and imbalance > 0.2:
                liquidity_confirm = True
            elif signal == 'sell' and imbalance < -0.2:
                liquidity_confirm = True

            # Decide to place an order only if both breakout confirmation and liquidity confirmation hold.
            trade_order = None
            if confirmed and liquidity_confirm and signal is not None:
                # Determine order quantity (base amount = 10) adjusted by available position space.
                base_order_qty = 10
                current_position = state.position.get(product, 0)
                pos_limit = 100  # assumed default position limit
                order_qty = base_order_qty
                if signal == 'buy':
                    available = pos_limit - current_position
                    order_qty = min(base_order_qty, available)
                elif signal == 'sell':
                    available = pos_limit + current_position  # for short positions
                    order_qty = min(base_order_qty, available)
                if order_qty > 0:
                    # Determine order price: use best available quote from the opposing side.
                    if signal == 'buy':
                        if order_depth.sell_orders:
                            price = min(order_depth.sell_orders.keys())
                        else:
                            price = midprice
                        trade_order = Order(product, price, order_qty)
                    elif signal == 'sell':
                        if order_depth.buy_orders:
                            price = max(order_depth.buy_orders.keys())
                        else:
                            price = midprice
                        trade_order = Order(product, price, -order_qty)

            orders = []
            if trade_order:
                orders.append(trade_order)

            orders_result[product] = orders

        # Serialize the updated persistent data back into traderData.
        traderData = jsonpickle.encode(data)
        conversions = 0  # No conversion requests in this strategy.
        result = orders_result
        return result, conversions, traderData