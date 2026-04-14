# MAG-7-ETF-Arbitrage-Strategy
A statistical arbitrage backtest that exploits mispricings between a synthetic Magnificent 7 ETF and its theoretical net asset value (NAV).

## How It Works

1. **NAV Construction** — A weighted basket of 7 stocks (MSFT, AAPL, META, AMZN, GOOGL, NVDA, TSLA) forms the theoretical fair value. Weights are fixed and sum to 1.

2. **ETF Price** — Simulated as the NAV plus a mean-reverting AR(1) mispricing process. This models the real-world phenomenon where ETF market prices temporarily deviate from their underlying value.

3. **Signal Generation** — When the percentage spread (ETF vs NAV) exceeds an entry threshold, the strategy enters a position:
   - ETF overpriced vs NAV → short ETF, long basket
   - ETF underpriced vs NAV → long ETF, short basket
   - Position exits when the spread reverts near zero

4. **PnL & Costs** — Daily PnL is computed from spread changes. Transaction costs (0.005% per trade) are deducted on every entry/exit.

## Data

Real historical closing prices are pulled via `yfinance` for the 2024 calendar year (~251 trading days). The mispricing layer is synthetic (AR(1) noise), since no actual MAG-7 ETF exists to source real spread data from.

## Quick Start

```bash
pip install numpy pandas matplotlib yfinance
python3 etf_arb.py
```

## Output

- Spread statistics (mean, std, min/max)
- Performance metrics: total PnL, Sharpe ratio, max drawdown, trade count, cost drag
- 4-panel chart: ETF vs NAV, % spread with thresholds, position timeline, cumulative PnL

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `phi` | 0.85 | AR(1) persistence of mispricing |
| `ENTRY_THRESHOLD` | 0.02% | Spread level to enter a trade |
| `EXIT_THRESHOLD` | 0.005% | Spread level to exit a trade |
| `COST_PER_TRADE` | 0.005% | Transaction cost per round trip |
