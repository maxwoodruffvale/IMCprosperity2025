# Competition Overview

The **IMC Prosperity Trading Competition** is an algorithmic trading challenge where participants develop automated trading strategies to compete in a simulated market environment. The goal is to maximize profit by intelligently buying, selling, and converting various financial instruments (products) based on real-time market data. The competition progresses through several rounds, each introducing new products and market complexities, requiring adaptive and robust trading algorithms.

---

## Our Team (Team "B")

**Max Woodruff Vale, Derek Days, Tuyen Nguyen, Alex Kumar, Brendan Lee**

---

## Results: **1.076.599 Shells/Currency**

- **USA:** #5  
- **International:** #17

---

## Round-by-Round Algorithmic Strategy Breakdown

---

### **Round 1: Basic Market Making and Mean Reversion**

**Summary of Algorithms Used:**

- **Market Making:**  
  For `RAINFOREST_RESIN` and `KELP`, a basic market-making strategy was employed. This involved placing bid and ask orders around a calculated fair value, aiming to profit from the bid-ask spread.

- **Mean Reversion (Z-Score based):**  
  For `SQUID_INK`, a mean-reversion strategy analyzed the Z-score of the current price relative to a rolling average. Trades were made to revert to the mean when the deviation was significant.

- **Clear Orders:**  
  A `clear_orders` helper function managed positions by placing zero-profit orders to reduce inventory risk.

**Changes Made:**

- Implemented basic market-making logic for `resin` and `kelp`.
- Added Z-score-based mean reversion for `ink`.
- Used `jsonpickle` for `traderData` persistence.

**New Introductions:**

- `RAINFOREST_RESIN`: Fixed fair value of 10000.  
- `KELP`: Fair value from mid-price of highest bid and lowest ask.  
- `SQUID_INK`: Mean-reverting product.  
- `clear_orders`: Utility for inventory management.

---

### **Round 2: Basket Arbitrage**

**Summary of Algorithms Used:**

- **Basket Arbitrage:**  
  Focused on mispricings between `PICNIC_BASKET`s and components.

  - **Synthetic 1 (`PICNIC_BASKET1`)** = 6 × `CROISSANTS`, 3 × `JAMS`, 1 × `DJEMBES`  
  - **Synthetic 2 (`PICNIC_BASKET2`)** = 4 × `CROISSANTS`, 2 × `JAMS`  
  - **Synthetic 3 (`DJEMBES`)**: Arbitrage with both baskets.

- **Z-Score on Spreads:**  
  Z-scores on spread between real basket price and synthetic value to identify arbitrage.

- **Market Taking:**  
  Market orders executed on both sides to capture arbitrage spread.

**Changes Made:**

- Refined `ink_strategy` parameters.  
- Added synthetic depth functions and strategies for each basket.  
- Built `spread_picker` for optimal arbitrage targeting.

**New Introductions:**

- `CROISSANTS`, `JAMS`, `DJEMBES`: New components.  
- `PICNIC_BASKET1`, `PICNIC_BASKET2`: Composite products.  
- Multi-product arbitrage logic.  
- `Logger` class for debugging.

---

### **Round 3: Options Trading (Black-Scholes Implied Volatility)**

**Summary of Algorithms Used:**

- **Black-Scholes Model:**  
  Used to price European call options (`VOLCANIC_ROCK_VOUCHER_X`).

- **Implied Volatility (IV):**  
  `find_vol` (Newton-Raphson method) calculated IV from market price.

- **IV Mean Reversion:**  
  Reversion-based strategy on IV Z-scores, like `SQUID_INK`.

**Changes Made:**

- Added `bs_call` and `bs_vega` for calculations.  
- Implemented `voucher_strategy` per strike.  
- Tuned Z-score windows and thresholds.  
- Stored basket `fair_value` to avoid recomputation.

**New Introductions:**

- `VOLCANIC_ROCK`: Option underlying.  
- Options:  
  - `VOLCANIC_ROCK_VOUCHER_9500`  
  - `VOLCANIC_ROCK_VOUCHER_9750`  
  - `VOLCANIC_ROCK_VOUCHER_10000`  
  - `VOLCANIC_ROCK_VOUCHER_10250`  
  - `VOLCANIC_ROCK_VOUCHER_10500`

---

### **Round 4: Hardcoded Voucher Strategy and Macarons Conversion**

**Summary of Algorithms Used:**

- **Hardcoded Strategy:**  
  For `VOLCANIC_ROCK_VOUCHER_10500`, simple fixed-threshold logic was applied.

- **Macarons Conversion:**  
  Converted `MAGNIFICENT_MACARONS` using prices, fees, and tariffs in a direct arbitrage strategy.

**Changes Made:**

- Added `voucher_hard_strategy` with fixed thresholds.  
- Built `macarons_strategy` for product conversion.  
- Adjusted T (time to expiry) formula to `(7 - timestamp / 10000) / 365`.

**New Introductions:**

- `MAGNIFICENT_MACARONS`: Convertible product.  
- `voucher_hard_strategy`: Simpler, robust logic for a single option.

---

### **Round 5: Olivia Interaction and Internal Position Tracking**

**Summary of Algorithms Used:**

- **Olivia Interaction:**  
  `ink_strategy` and `picnic_olivia_strategy` reacted to trades by "Olivia", a signal-rich trader.

- **Internal Position Tracking:**  
  `internal_position` dictionary tracks holdings in related products for safer arbitrage.

- **Resuming Arbitrage:**  
  `resume_picnic_arb` flag controls pause/resume of basket arbitrage based on Olivia's activity.

**Changes Made:**

- Added `self.ink_olivia` and `self.picnic_olivia` to track Olivia's behavior.  
- Developed `picnic_olivia_strategy`, `picnic_olivia_helper`, and `correlated_picnic_olivia_helper`.  
- Enhanced real-time inventory tracking with `internal_position`.  
- Introduced `add_to_dict_list` for order management.  
- Further adjusted T formula to `(6 - timestamp / 10000) / 365`.

**New Introductions:**

- Real-time strategy adaptation to a specific trader (`Olivia`).  
- Fine-grained internal tracking for multi-leg arbitrage.  
- Context-sensitive strategy switching.
