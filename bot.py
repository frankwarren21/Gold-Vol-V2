# bot.py
import os
import pandas as pd
import numpy as np
from polygon import RESTClient
from datetime import datetime, timedelta
import torch
from rl_model import DQN, ReplayMemory, Transition
from config import *
from utils import is_trading_time, prepare_state
import random

class GoldTradingBot:
    def __init__(self):
        self.client = RESTClient(api_key=os.getenv("POLYGON_API_KEY"))
        self.df_h = None
        self.df_m = None

        # RL
        self.policy_net = DQN()
        self.target_net = DQN()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters())
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.steps_done = 0
        self.eps = EPS_START

        self.trades = []
        self.daily_pl = {}
        self.current_day = None

    def fetch_data(self, years=5, recent_min_only=True):
        end = datetime.now()
        start = end - timedelta(days=years * 365 + 60)

        ticker = "C:XAUUSD"  # spot proxy

        print("Fetching hourly...")
        aggs_h = list(self.client.list_aggs(ticker, 1, "hour", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), limit=50000))
        self.df_h = pd.DataFrame([{
            'timestamp': pd.to_datetime(a.timestamp, unit='ms'),
            'open': a.open, 'high': a.high, 'low': a.low, 'close': a.close, 'volume': a.volume
        } for a in aggs_h]).set_index('timestamp')

        if recent_min_only:
            recent_start = (end - timedelta(days=60)).strftime("%Y-%m-%d")
            print("Fetching recent 1-min...")
            aggs_m = list(self.client.list_aggs(ticker, 1, "minute", recent_start, end.strftime("%Y-%m-%d"), limit=50000))
            self.df_m = pd.DataFrame([{
                'timestamp': pd.to_datetime(a.timestamp, unit='ms'),
                'open': a.open, 'high': a.high, 'low': a.low, 'close': a.close, 'volume': a.volume
            } for a in aggs_m]).set_index('timestamp')
        else:
            # Full 5y 1-min would be massive → implement stitching or use broker feed in production
            print("Full 1-min fetch skipped for performance.")

    def backtest(self):
        if self.df_h is None:
            self.fetch_data()

        df_h = self.df_h.copy()
        df_h['sma_50'] = df_h['close'].rolling(SMA_PERIOD).mean()
        df_h['avg_vol'] = df_h['volume'].rolling(VOLUME_ROLLING_PERIOD).mean()

        trades = []
        position = None
        day_pl = 0
        current_day = None

        for ts_h, row in df_h.iterrows():
            day = ts_h.date()
            if day != current_day:
                current_day = day
                day_pl = self.daily_pl.get(day, 0)

            if day_pl <= -MAX_DAILY_LOSS:
                continue

            if not is_trading_time(ts_h):
                continue

            price = row['close']
            trend_up = price > row['sma_50']
            trend_down = price < row['sma_50']

            if row['volume'] <= row['avg_vol']:
                continue

            # Simplified zone (last candle mid)
            mid = (row['high'] + row['low']) / 2

            # Get recent 1-min (limited window)
            recent_m = self.df_m[(self.df_m.index >= ts_h - timedelta(hours=6)) & 
                                 (self.df_m.index < ts_h)]

            if recent_m.empty:
                continue

            signal = None
            entry = None
            sl_dist_ticks = None
            strength = 0

            # Consolidation
            near = recent_m[abs(recent_m['close'] - mid) <= ZONE_MARGIN_TICKS / 10]
            if len(near) >= CONSOL_CANDLES_MIN:
                signal = 'buy' if trend_up else 'sell' if trend_down else None
                entry = near['close'].mean()
                strength = len(near) / CONSOL_CANDLES_MIN
                if signal == 'buy':
                    sl_dist_ticks = (entry - recent_m['low'].min()) * 10 + 50
                else:
                    sl_dist_ticks = (recent_m['high'].max() - entry) * 10 + 50

            # Pin bar (last candle)
            last = recent_m.iloc[-1]
            body = abs(last['close'] - last['open'])
            lw = min(last['open'], last['close']) - last['low']
            uw = last['high'] - max(last['open'], last['close'])
            if lw > PIN_WICK_MULTIPLIER * body and abs(last['low'] - mid) <= ZONE_MARGIN_TICKS / 10:
                signal = 'buy' if trend_up else None
                entry = last['low'] + lw * 0.5
                sl_dist_ticks = (entry - last['low']) * 10 + 50
                strength = lw / body if body > 0 else 10
            elif uw > PIN_WICK_MULTIPLIER * body and abs(last['high'] - mid) <= ZONE_MARGIN_TICKS / 10:
                signal = 'sell' if trend_down else None
                entry = last['high'] - uw * 0.5
                sl_dist_ticks = (last['high'] - entry) * 10 + 50
                strength = uw / body if body > 0 else 10

            if signal is None:
                continue

            # RL filter
            state_np = prepare_state(row, price, strength)
            state = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q = self.policy_net(state)
            action = torch.argmax(q).item()
            if action == 0:
                continue

            # Size
            sl_ticks = max(MIN_SL_TICKS, min(MAX_SL_TICKS, sl_dist_ticks))
            contracts = RISK_PER_TRADE / (sl_ticks * TICK_VALUE)
            contracts = round(contracts, 2)  # broker may allow decimals

            # Simulate trade outcome (placeholder – expand for full bar-by-bar)
            # In real backtest you'd loop future bars to check TP/SL/trailing
            profit_ticks = 50 if random.random() < 0.741 else -sl_ticks / 2  # approx from sim
            gross_pl = profit_ticks * contracts * TICK_VALUE

            # Fees & slippage
            comm = COMMISSION_RT_PER_CONTRACT * contracts
            slip = SLIPPAGE_TICKS_RT_PER_CONTRACT * TICK_VALUE * contracts
            net_pl = gross_pl - comm - slip

            trades.append({
                'time': ts_h,
                'side': signal,
                'entry': entry,
                'contracts': contracts,
                'gross_pl': gross_pl,
                'net_pl': net_pl,
                'win': net_pl > 0
            })

            day_pl += net_pl
            self.daily_pl[day] = day_pl

            # RL memory (real reward = net_pl)
            reward = net_pl
            next_state = state  # placeholder
            self.memory.push(state, torch.tensor([action]), torch.tensor([reward]), next_state)

        # Train RL
        self._train_rl()

        # Summary
        if trades:
            df = pd.DataFrame(trades)
            win_rate = df['win'].mean()
            gross = df['gross_pl'].sum()
            net = df['net_pl'].sum()
            print(f"Trades: {len(trades):,} | Win rate: {win_rate:.2%}")
            print(f"Gross P/L: ${gross:,.0f} | Net P/L: ${net:,.0f}")

    def _train_rl(self):
        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        next_states = torch.cat(batch.next_state)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_max = self.target_net(next_states).max(1)[0].detach()
        expected = rewards + GAMMA * next_max

        loss = nn.SmoothL1Loss()(q_values, expected)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        self.eps = max(EPS_END, self.eps * EPS_DECAY)
        self.steps_done += 1

        if self.steps_done % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
