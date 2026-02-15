# config.py
"""All configurable parameters for the MGC trading bot"""

# ────────────────────────────────────────────────
# ACCOUNT & RISK
# ────────────────────────────────────────────────
ACCOUNT_SIZE = 50000
RISK_PER_TRADE = 500          # $ max risk per trade
MAX_DAILY_LOSS = 1000         # $ (2% of account)

# ────────────────────────────────────────────────
# MGC CONTRACT SPECS
# ────────────────────────────────────────────────
TICK_VALUE = 1.0              # $ per tick per contract
MIN_SL_TICKS = 100
MAX_SL_TICKS = 500
TP_TICKS = 50                 # locked-in final target
TRAIL_ACTIVATE_TICKS = 20     # locked-in final activation
TRAIL_DISTANCE_TICKS = 40

ZONE_MARGIN_TICKS = 50        # ±5 points
CONSOL_CANDLES_MIN = 10
PIN_WICK_MULTIPLIER = 4

# ────────────────────────────────────────────────
# FILTERS & WINDOWS
# ────────────────────────────────────────────────
SMA_PERIOD = 50
VOLUME_ROLLING_PERIOD = 20

TRADING_START_HOUR = 7
TRADING_START_MIN = 30
TRADING_END_HOUR = 15
TRADING_END_MIN = 30

# ────────────────────────────────────────────────
# RL
# ────────────────────────────────────────────────
STATE_SIZE = 4
ACTION_SIZE = 2               # 0=skip, 1=take
HIDDEN_SIZE = 64
GAMMA = 0.95
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
BATCH_SIZE = 32
MEMORY_SIZE = 2000
TARGET_UPDATE = 100

# ────────────────────────────────────────────────
# FEES & SLIPPAGE (realistic)
# ────────────────────────────────────────────────
COMMISSION_RT_PER_CONTRACT = 0.50     # round-trip
SLIPPAGE_TICKS_RT_PER_CONTRACT = 1    # 1 tick round-trip average
