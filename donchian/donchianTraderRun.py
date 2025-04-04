import jsonpickle
import numpy as np
import pandas as pd
import math
from typing import List, Dict
import statistics
from datamodel import TradingState, Order, OrderDepth, Trade

class Trader:
    def __init__(self):
        pass

    def run(self, state: TradingState):
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

        for product, order_depth in state.order_depths.items():
            # Compute midprice from best bid and best ask if possible
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
                continue  # No price info available

            # Initialize price history for this product if not present.
            if product not in data['price_history']:
                data['price_history'][product] = []
            # Append midprice only if it differs from the last one.
            if not data['price_history'][product] or data['price_history'][product][-1] != midprice:
                data['price_history'][product].append(midprice)
            # Keep only the latest 200 prices
            if len(data['price_history'][product]) > 200:
                data['price_history'][product] = data['price_history'][product][-200:]
            
            price_series = data['price_history'][product]

            # Calculate volatility from price differences (if possible)
            if len(price_series) >= 2:
                returns = np.diff(price_series)
                volatility = np.std(returns)
            else:
                volatility = 0.0

            # Determine optimal lookback period based on volatility.
            epsilon = 1e-5
            optimal_lookback = int(np.clip(50 / (volatility + epsilon), 10, 100))
            # Exclude the current price from the window for breakout calculation if possible.
            if len(price_series) > 1:
                window = price_series[-optimal_lookback:-1] if len(price_series) >= optimal_lookback + 1 else price_series[:-1]
            else:
                window = price_series

            # If we have a window, calculate rolling high and low
            if window:
                rolling_high = max(window)
                rolling_low = min(window)
            else:
                rolling_high = midprice
                rolling_low = midprice

            # Initialize breakout confirmation storage for this product if not present.
            if product not in data['breakout_confirmations']:
                data['breakout_confirmations'][product] = {'signal': None, 'count': 0}

            # Determine breakout signal using a threshold of 0.5%
            signal = None
            if window and midprice > rolling_high * 1.005:
                signal = 'buy'
            elif window and midprice < rolling_low * 0.995:
                signal = 'sell'

            # Update confirmation count: require two consecutive signals.
            if signal == data['breakout_confirmations'][product]['signal'] and signal is not None:
                data['breakout_confirmations'][product]['count'] += 1
            else:
                data['breakout_confirmations'][product]['signal'] = signal
                data['breakout_confirmations'][product]['count'] = 1

            confirmed = data['breakout_confirmations'][product]['count'] >= 2

            # Calculate order book imbalance.
            total_buy = sum(order_depth.buy_orders.values()) if order_depth.buy_orders else 0
            total_sell = abs(sum(order_depth.sell_orders.values())) if order_depth.sell_orders else 0
            imbalance = 0
            if (total_buy + total_sell) > 0:
                imbalance = (total_buy - total_sell) / (total_buy + total_sell)

            # Set liquidity confirmation thresholds.
            liquidity_confirm = False
            if signal == 'buy' and imbalance > 0.2:
                liquidity_confirm = True
            elif signal == 'sell' and imbalance < -0.2:
                liquidity_confirm = True

            trade_order = None
            # Primary condition: breakout confirmed and liquidity confirmed.
            if confirmed and liquidity_confirm and signal is not None:
                trade_order = self.create_order(signal, product, order_depth, state.position.get(product, 0))
            else:
                # Fallback: if insufficient price history or no confirmed breakout,
                # and there is moderate buying pressure, then issue a buy.
                if (len(price_series) < 5 or signal is None) and imbalance > 0.1:
                    trade_order = self.create_order('buy', product, order_depth, state.position.get(product, 0))

            orders = []
            if trade_order:
                orders.append(trade_order)
            orders_result[product] = orders

        # Serialize persistent data back into traderData.
        traderData = jsonpickle.encode(data)
        conversions = 0  # No conversion requests in this strategy.
        return orders_result, conversions, traderData

    def create_order(self, signal: str, product: str, order_depth: OrderDepth, current_position: int):
        # Base order quantity and assumed position limit.
        base_order_qty = 10
        pos_limit = 100  
        order_qty = base_order_qty

        if signal == 'buy':
            available = pos_limit - current_position
            order_qty = min(base_order_qty, available)
            if order_qty <= 0:
                return None
            # Use best ask for buy orders.
            if order_depth.sell_orders:
                price = min(order_depth.sell_orders.keys())
            else:
                price = 100  # fallback price
            return Order(product, price, order_qty)
        elif signal == 'sell':
            available = pos_limit + current_position  # for short positions
            order_qty = min(base_order_qty, available)
            if order_qty <= 0:
                return None
            # Use best bid for sell orders.
            if order_depth.buy_orders:
                price = max(order_depth.buy_orders.keys())
            else:
                price = 100  # fallback price
            return Order(product, price, -order_qty)
        return None

# Test harness with synthetic data
def generate_synthetic_prices(start_price=100, steps=50, volatility=2):
    np.random.seed(42)  # For reproducibility
    return np.cumsum(np.random.randn(steps) * volatility + start_price).tolist()

def generate_trading_state(prices, iteration):
    product = "APPLE"
    price = prices[iteration]
    order_depth = OrderDepth()
    order_depth.sell_orders = {price + 1: 5, price + 2: 10}  # Simulated sell orders
    order_depth.buy_orders = {price - 1: 5, price - 2: 10}  # Simulated buy orders
    
    market_trades = {
        product: [Trade(symbol=product, price=price, quantity=10)]
    }
    own_trades = {
        product: []
    }
    position = {product: 0}
    observations = {}
    # Initialize traderData with current history (simulate no duplicate current price)
    traderData = jsonpickle.encode({"price_history": {product: prices[:iteration]}, "breakout_confirmations": {product: {"signal": None, "count": 0}}})
    
    return TradingState(traderData, iteration, {}, {product: order_depth}, own_trades, market_trades, position, observations)

def test_algorithm(trader):
    prices = generate_synthetic_prices()
    for iteration in range(len(prices)):
        state = generate_trading_state(prices, iteration)
        orders, conversions, traderData = trader.run(state)
        print(f"Iteration {iteration}:")
        print("Orders:", orders)
        print("Conversions:", conversions)
        print("New traderData:", traderData)
        print("-" * 50)

# Example usage:
if __name__ == "__main__":
    trader = Trader()
    test_algorithm(trader)