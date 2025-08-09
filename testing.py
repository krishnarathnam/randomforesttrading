
import numpy as np
import pandas as pd


def _compute_atr(df, period=14):
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['close'].astype(float)
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = np.nanmax(np.vstack([tr1.values, tr2.values, tr3.values]), axis=0)
    atr = pd.Series(tr, index=df.index).rolling(period, min_periods=1).mean()
    return atr


def calculate_target_variable(df, barsupfront=3, SLTPRatio=2.0, n1=2, n2=2, atr_period=14, sl_mult=1.0):
    """
    Improved target variable using ATR-based dynamic barriers (triple-barrier style).

    - Uses ATR to size SL and TP distances instead of prior bar extremes.
    - For each signal, finds the first barrier hit within the horizon (barsupfront).
    - If both barriers are touched within a single bar, labels as 3 (both hit).

    Returns:
      amount: realized move from entry to exit level (price units; positive if favorable, negative if adverse)
      trendcat: categorical outcome per your convention:
        Bearish (signal==1): 1 = TP (win), 2 = SL (loss), 3 = both, 0 = no hit
        Bullish (signal==2): 2 = TP (win), 1 = SL (loss), 3 = both, 0 = no hit
    """
    # Ensure numeric types
    for col in ['high', 'low', 'close', 'open']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    length = len(df)
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    signal = df['signal'].values

    # ATR-based distances
    atr = _compute_atr(df, period=atr_period).values

    trendcat = [0] * length
    amount = [0.0] * length

    for line in range(n1, length - barsupfront - n2):
        side = signal[line]
        if side == 0 or np.isnan(close[line]) or np.isnan(atr[line]):
            continue

        # Distance in price units
        sl_distance = max(atr[line], 1e-8) * float(sl_mult)
        tp_distance = float(SLTPRatio) * sl_distance

        entry = close[line]

        if side == 1:  # Bearish: want down move
            sl_level = entry + sl_distance
            tp_level = entry - tp_distance
        elif side == 2:  # Bullish: want up move
            sl_level = entry - sl_distance
            tp_level = entry + tp_distance
        else:
            continue

        outcome_set = False
        for i in range(1, barsupfront + 1):
            idx = line + i
            if idx >= length:
                break

            bar_high = high[idx]
            bar_low = low[idx]
            if np.isnan(bar_high) or np.isnan(bar_low):
                continue

            hit_tp = (bar_high >= tp_level) if side == 2 else (bar_low <= tp_level)
            hit_sl = (bar_low <= sl_level) if side == 2 else (bar_high >= sl_level)

            if hit_tp and hit_sl:
                # Ambiguous within a single bar; mark as both
                trendcat[line] = 3
                amount[line] = 0.0
                outcome_set = True
                break
            elif hit_tp:
                if side == 1:
                    trendcat[line] = 1  # bearish TP
                    amount[line] = entry - tp_level
                else:
                    trendcat[line] = 2  # bullish TP
                    amount[line] = tp_level - entry
                outcome_set = True
                break
            elif hit_sl:
                if side == 1:
                    trendcat[line] = 2  # bearish SL
                    amount[line] = entry - sl_level
                else:
                    trendcat[line] = 1  # bullish SL
                    amount[line] = sl_level - entry
                outcome_set = True
                break

        # If neither barrier hit within horizon, keep 0 and amount 0.0

    return amount, trendcat
