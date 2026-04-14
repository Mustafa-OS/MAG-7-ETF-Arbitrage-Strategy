import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

tickers = ['MSFT', 'AAPL', 'META', 'AMZN', 'GOOGL', 'NVDA', 'TSLA']

# ETF weights (must sum to 1)
weights = {
    'MSFT': 0.20, 'AAPL': 0.20, 'META': 0.15,
    'AMZN': 0.15, 'GOOGL': 0.15, 'NVDA': 0.10, 'TSLA': 0.05
}

# ~1 trading year of data
data = yf.download(tickers, start='2024-01-01', end='2024-12-31')['Close']
data = data.dropna()

df = data.copy()


# NAV = weighted sum of underlying stock prices
df['NAV'] = sum(weights[t] * df[t] for t in tickers)


# ETF price = NAV + mean-reverting mispricing (AR(1) process)
# AR(1) = autocorrelated noise, more realistic than pure random noise
# phi controls persistence: how long a mispricing lasts before closing
np.random.seed(42)
n = len(df)
phi = 0.85
mispricing = np.zeros(n)
for i in range(1, n):
    mispricing[i] = phi * mispricing[i-1] + np.random.randn() * 0.05

df['ETF'] = df['NAV'] + mispricing


# % spread = how far ETF is from NAV, as a percentage of NAV
# Positive -> ETF overpriced relative to NAV
# Negative -> ETF underpriced relative to NAV
df['pct_spread'] = (df['ETF'] - df['NAV']) / df['NAV'] * 100

print("Spread stats (%):")
print(df['pct_spread'].describe().round(4))


# Entry: trade when spread exceeds threshold (in %)
# Exit:  close when spread returns close to 0
ENTRY_THRESHOLD = 0.02   # 0.02% mispricing -> enter
EXIT_THRESHOLD  = 0.005  # 0.005% -> close (near enough to fair value)

# Position:
#  +1 = Long ETF, Short basket (ETF underpriced)
#  -1 = Short ETF, Long basket (ETF overpriced)
#   0 = Flat

position  = 0
positions = []

for spread in df['pct_spread']:
    if position == 0:
        if spread > ENTRY_THRESHOLD:
            position = -1       # ETF too expensive -> short it
        elif spread < -ENTRY_THRESHOLD:
            position = 1        # ETF too cheap -> buy it

    elif position == 1 and spread >= -EXIT_THRESHOLD:
        position = 0            # spread closed -> exit long

    elif position == -1 and spread <= EXIT_THRESHOLD:
        position = 0            # spread closed -> exit short

    positions.append(position)

df['position'] = positions


# Daily PnL = yesterday's position x today's change in spread
# (we trade the spread, so profit = spread narrowing in our favour)
df['spread_chg'] = df['pct_spread'].diff()
df['pnl_daily']  = df['position'].shift(1) * df['spread_chg']
df['pnl_cum']    = df['pnl_daily'].cumsum()


COST_PER_TRADE = 0.005  # 0.005% per trade (realistic for ETF arb)

df['trade']      = df['position'].diff().abs()   # 1 on days we trade
df['pnl_daily'] -= df['trade'] * COST_PER_TRADE
df['pnl_cum']    = df['pnl_daily'].cumsum()      # recalculate after costs


sharpe   = df['pnl_daily'].mean() / df['pnl_daily'].std() * np.sqrt(252)
max_dd   = (df['pnl_cum'] - df['pnl_cum'].cummax()).min()
n_trades = int(df['trade'].sum())
total_costs = (df['trade'] * COST_PER_TRADE).sum()

print(f"\n{'─'*35}")
print(f"  Total PnL (% spread):  {df['pnl_cum'].iloc[-1]:.4f}%")
print(f"  Sharpe Ratio:          {sharpe:.2f}")
print(f"  Max Drawdown:          {max_dd:.4f}%")
print(f"  Number of Trades:      {n_trades}")
print(f"  Total Cost Drag:       {total_costs:.4f}%")
print(f"{'─'*35}")



fig, axes = plt.subplots(4, 1, figsize=(10, 8))
fig.suptitle('MAG-7 ETF Arbitrage Strategy', fontsize=14, fontweight='bold')

# Plot 1: ETF vs NAV
axes[0].plot(df['ETF'].values, label='ETF Price',         linewidth=1.2)
axes[0].plot(df['NAV'].values, label='Theoretical NAV',   linewidth=1.2, linestyle='--')
axes[0].set_title('ETF Price vs Theoretical NAV')
axes[0].legend()

# Plot 2: Percentage Spread
axes[1].plot(df['pct_spread'].values, color='purple', linewidth=1)
axes[1].axhline( ENTRY_THRESHOLD, color='red',   linestyle='--', label=f'Short ETF (+{ENTRY_THRESHOLD}%)')
axes[1].axhline(-ENTRY_THRESHOLD, color='green', linestyle='--', label=f'Long ETF (-{ENTRY_THRESHOLD}%)')
axes[1].axhline(0, color='black', linewidth=0.8)
axes[1].set_title('% Spread (ETF - NAV) / NAV')
axes[1].legend()

# Plot 3: Position
axes[2].plot(df['position'].values, color='steelblue', drawstyle='steps-post', linewidth=1)
axes[2].set_title('Position: +1 = Long ETF | -1 = Short ETF | 0 = Flat')
axes[2].set_yticks([-1, 0, 1])
axes[2].set_yticklabels(['Short ETF', 'Flat', 'Long ETF'])

# Plot 4: Cumulative PnL
axes[3].plot(df['pnl_cum'].values, color='green', linewidth=1.2)
axes[3].axhline(0, color='black', linewidth=0.8)
axes[3].fill_between(range(len(df)), df['pnl_cum'].values, 0,
                     where=df['pnl_cum'].values >= 0, alpha=0.2, color='green')
axes[3].fill_between(range(len(df)), df['pnl_cum'].values, 0,
                     where=df['pnl_cum'].values < 0,  alpha=0.2, color='red')
axes[3].set_title('Cumulative PnL (% spread, after transaction costs)')

plt.tight_layout()
plt.show()
