# Gold MGC Day Trading Bot

Locked configuration (February 2026):
- Micro Gold Futures (MGC)
- TP: 50 ticks
- Trailing stop: activate +20 ticks, trail 40 ticks
- Risk: $500 per trade
- Realistic fees & slippage included

## Setup

1. `pip install -r requirements.txt`
2. Copy `.env.example` → `.env` and add your Polygon API key
3. `python main.py`

## Notes

- Backtest uses Polygon spot proxy (XAUUSD)
- 1-min data limited to recent 60 days (expand or use broker feed for full history)
- Live trading requires broker integration (Tradovate/IBKR/etc.)

For questions or improvements → open an issue.
