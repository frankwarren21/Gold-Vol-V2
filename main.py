# main.py
from bot import GoldTradingBot

if __name__ == "__main__":
    print("Gold MGC Trading Bot - Locked Config (Feb 2026)")
    print("TP: 50 ticks | Trail start: 20 ticks | Trail dist: 40 ticks")
    print("Risk: $500/trade | Fees & slippage included\n")

    bot = GoldTradingBot()
    bot.fetch_data()
    bot.backtest()

    print("\nDone.")
