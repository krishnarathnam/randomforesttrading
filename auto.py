
import numpy as np
import pandas as pd
import mplfinance as mpf
from dotenv import load_dotenv
from twelvedata import TDClient
import os
import yfinance as yf
from testing import calculate_target_variable
import joblib 

df = yf.download("AAPL", period="20y")
df.reset_index(inplace=True)
df = df.iloc[2:] 
df['Date'] = pd.to_datetime(df['Date'])
df.columns = ['date', 'close', 'high', 'low', 'open', 'volume']
df.set_index('date', inplace=True)


def calculateSR(df,n1=2,n2=2,threshold_ratio=0.0010):
    def support(df1,l,n1,n2):
        for i in range(l-n1+1,l+1):
            if(df1.iloc[i]['low']>df1.iloc[i-1]['low']):
                return 0
        for i in range(l+1,l+n2+1):
            if(df1.iloc[i]['low']<df1.iloc[i-1]['low']):
                return 0
        return 1

    def resistance(df1,l,n1,n2):
        for i in range(l-n1+1,l+1):
            if(df1.iloc[i]['high']<df1.iloc[i-1]['high']):
                return 0
        for i in range(l+1,l+n2+1):
            if(df1.iloc[i]['high']>df1.iloc[i-1]['high']):
                return 0
        return 1

    sr = []

    for row in range(n1,len(df)-n2):
        if support(df,row,n1,n2):
            sr.append((row,float(df.iloc[row]['low']),1)) # support
        if resistance(df,row,n1,n2):
            sr.append((row,float(df.iloc[row]['high']),2)) # resistance

    def filter_nearby_levels(levels, threshold_ratio):
        if not levels:
            return []

        levels = sorted(levels)
        filtered = [levels[0]]
        for lvl in levels[1:]:
            if abs(lvl - filtered[-1]) >= threshold_ratio * lvl:
                filtered.append(lvl)
        return filtered
    support = filter_nearby_levels([x[1] for x in sr if x[2] == 1], threshold_ratio)  # supports
    resistance = filter_nearby_levels([x[1] for x in sr if x[2] == 2], threshold_ratio) # resistance

    return support,resistance


def nearest_level(price,levels):
    if not levels:
        return None
    else:
        return min(levels, key=lambda x: abs(x-price))


def Revsignal1(df1):
    length = len(df1)
    high = list(df1['high'])
    low = list(df1['low'])
    close = list(df1['close'])
    open = list(df1['open'])
    signal = [0]*length
    bodydiff = [0]*length

    for row in range(1, length):
            bodydiff[row] = abs(open[row]-close[row])
            bodydiffmin = 0.003
            if (bodydiff[row]>bodydiffmin and bodydiff[row-1]>bodydiffmin and
                open[row-1]<close[row-1] and
                open[row]>close[row] and 
                #open[row]>=close[row-1] and close[row]<open[row-1]):
                (open[row]-close[row-1])>=+0e-5 and close[row]<open[row-1]):
                signal[row] = 1
            elif (bodydiff[row]>bodydiffmin and bodydiff[row-1]>bodydiffmin and
                open[row-1]>close[row-1] and
                open[row]<close[row] and 
                #open[row]<=close[row-1] and close[row]>open[row-1]):
                (open[row]-close[row-1])<=-0e-5 and close[row]>open[row-1]):
                signal[row] = 2
            else:
                signal[row] = 0
    return signal




def Revsignal2(df):

    length = len(df)
    high = list(df['high'])
    low = list(df['low'])
    close = list(df['close'])
    open = list(df['open'])
    signal = [0]*length
    bodydiff = [0]*length
    highdiff = [0]*length
    lowdiff = [0]*length
    ratio1 = [0]*length
    ratio2 = [0]*length

    for row in range(0, length):

        highdiff[row] = high[row]-max(open[row],close[row])
        bodydiff[row] = abs(open[row]-close[row])
        if bodydiff[row]<0.002:
            bodydiff[row]=0.002
        lowdiff[row] = min(open[row],close[row])-low[row]
        ratio1[row] = highdiff[row]/bodydiff[row]
        ratio2[row] = lowdiff[row]/bodydiff[row]

        
        if (ratio1[row]>2.0 and lowdiff[row]<0.4*highdiff[row] and bodydiff[row]>0.02 ): #and df.RSI[row]>60 and df.RSI[row]<80 
            signal[row] = 1
        
        #elif (ratio2[row-1]>2.5 and highdiff[row-1]<0.23*lowdiff[row-1] and bodydiff[row-1]>0.03 and bodydiff[row]>0.04 and close[row]>open[row] and close[row]>high[row-1] and df.RSI[row]<55 and df.RSI[row]>30):
        #    signal[row] = 2
        
        elif (ratio2[row]>2.5 and highdiff[row]<0.23*lowdiff[row] and bodydiff[row]>0.03 ):# and df.RSI[row]<55 and df.RSI[row]>10
            signal[row]=2
    return signal



def closeResistance(df, l, levels, lim):
    """Return 1 if candle l is a valid resistance rejection near the nearest level.

    Rules:
    - Proximity by percentage: near if high >= level OR |high-level|/level <= lim and high < level
    - Body must close/open below the level (no body above)
    - Prefer an upper-wick rejection (upper wick at least 30% of body). If no touch, wick rule is relaxed
    - Next candle must not break above the level (allow slight tolerance of lim)
    - Previous close should not be materially above the level (allow slight tolerance of lim)
    """
    if not levels:
        return 0

    level = min(levels, key=lambda x: abs(x - df['high'].iloc[l]))

    o = float(df['open'].iloc[l])
    c = float(df['close'].iloc[l])
    h = float(df['high'].iloc[l])
    lw = float(df['low'].iloc[l])

    body_high = max(o, c)
    body_low = min(o, c)
    body = max(abs(c - o), 1e-12)

    # Proximity as ratio to level
    rel_dist = abs(h - level) / max(level, 1e-12)
    near = (h >= level) or (rel_dist <= lim and h < level)

    # Body must be below the level
    body_below = body_high <= level

    # Wick rejection (if touched)
    upper_wick = max(h - body_high, 0.0)
    wick_ok = True if h < level else (upper_wick / body) >= 0.3

    # Previous candle context (avoid prior strong closes above)
    prev_ok = True
    if l - 1 >= 0:
        p_close = float(df['close'].iloc[l - 1])
        prev_ok = p_close <= level * (1 + lim)

    # Next candle confirmation (no break above)
    next_ok = True
    if l + 1 < len(df):
        n_o = float(df['open'].iloc[l + 1])
        n_c = float(df['close'].iloc[l + 1])
        n_h = float(df['high'].iloc[l + 1])
        next_ok = (max(n_o, n_c) <= level * (1 + lim)) and (n_h <= level * (1 + lim))

    return 1 if (near and body_below and wick_ok and prev_ok and next_ok) else 0
    
def closeSupport(df, l, levels, lim):
    """Return 1 if candle l is a valid support bounce near the nearest level.

    Rules:
    - Proximity by percentage: near if low <= level OR |low-level|/level <= lim and low > level
    - Body must close/open above the level (no body below)
    - Prefer a lower-wick bounce (lower wick at least 30% of body). If no touch, wick rule is relaxed
    - Next candle must not break below the level (allow slight tolerance of lim)
    - Previous close should not be materially below the level (allow slight tolerance of lim)
    """
    if not levels:
        return 0

    level = min(levels, key=lambda x: abs(x - df['low'].iloc[l]))

    o = float(df['open'].iloc[l])
    c = float(df['close'].iloc[l])
    h = float(df['high'].iloc[l])
    lw = float(df['low'].iloc[l])

    body_high = max(o, c)
    body_low = min(o, c)
    body = max(abs(c - o), 1e-12)

    # Proximity as ratio to level
    rel_dist = abs(lw - level) / max(level, 1e-12)
    near = (lw <= level) or (rel_dist <= lim and lw > level)

    # Body must be above the level
    body_above = body_low >= level

    # Wick bounce (if touched)
    lower_wick = max(body_low - lw, 0.0)
    wick_ok = True if lw > level else (lower_wick / body) >= 0.3

    # Previous candle context (avoid prior strong closes below)
    prev_ok = True
    if l - 1 >= 0:
        p_close = float(df['close'].iloc[l - 1])
        prev_ok = p_close >= level * (1 - lim)

    # Next candle confirmation (no break below)
    next_ok = True
    if l + 1 < len(df):
        n_o = float(df['open'].iloc[l + 1])
        n_c = float(df['close'].iloc[l + 1])
        n_l = float(df['low'].iloc[l + 1])
        next_ok = (min(n_o, n_c) >= level * (1 - lim)) and (n_l >= level * (1 - lim))

    return 1 if (near and body_above and wick_ok and prev_ok and next_ok) else 0


n1 = 2
n2 = 2
threshold = 0.006
df['engulfing'] = Revsignal1(df)
df['star'] = Revsignal2(df)
support, resistance = calculateSR(df,n1,n2,threshold)
hlines = support + resistance
hl_colors = ['green'] * len(support) + ['red'] * len(resistance)

df['nearest_support'] = df['low'].apply(lambda x: nearest_level(x, support))
df['nearest_resistance'] = df['high'].apply(lambda x: nearest_level(x, resistance))

df['dist_to_support'] = (df['low'] - df['nearest_support']) / df['nearest_support']
df['dist_to_resistance'] = (df['nearest_resistance'] - df['high']) / df['nearest_resistance']

df['signal'] = 0
for row in range(n1,len(df)-n2):
    # Make resistance proximity more lenient for bearish signals
    if((df['engulfing'].iloc[row]==1 or df['star'].iloc[row]==1) and closeResistance(df,row,resistance,0.0020)):
        df.loc[df.index[row], 'signal'] = 1
    elif((df['engulfing'].iloc[row]==2 or df['star'].iloc[row]==2) and closeSupport(df,row,support,0.0015)):
        df.loc[df.index[row], 'signal'] = 2
    else:
        df.loc[df.index[row], 'signal'] = 0


bearish_with_resistance = 0
bullish_with_support = 0
for row in range(n1,len(df)-n2):
    if df['engulfing'].iloc[row]==1 or df['star'].iloc[row]==1:
        if closeResistance(df,row,resistance,0.0015):
            bearish_with_resistance += 1
    if df['engulfing'].iloc[row]==2 or df['star'].iloc[row]==2:
        if closeSupport(df,row,support,0.0015):
            bullish_with_support += 1


signal = df['signal']
bull_marker = [np.nan] * len(df)
bear_marker = [np.nan] * len(df)
for i in range(len(df)):
    if signal.iloc[i] == 2:
        bull_marker[i] = df['low'].iloc[i] * 1
    elif signal.iloc[i] == 1:
        bear_marker[i] = df['high'].iloc[i] * 1

engulfing = df['engulfing']
engulfing_bull_marker = [np.nan] * len(df)
engulfing_bear_marker = [np.nan] * len(df)
for i in range(len(df)):
    if engulfing.iloc[i] == 2:
        engulfing_bull_marker[i] = df['low'].iloc[i] * 1
    elif engulfing.iloc[i] == 1:
        engulfing_bear_marker[i] = df['high'].iloc[i] * 1

star = df['star']
star_bull_marker = [np.nan] * len(df)
star_bear_marker = [np.nan] * len(df)
for i in range(len(df)):
    if star.iloc[i] == 2:
        star_bull_marker[i] = df['low'].iloc[i] * 1
    elif star.iloc[i] == 1:
        star_bear_marker[i] = df['high'].iloc[i] * 1
#

delta = df['close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()

rs = avg_gain / avg_loss
df['rsi'] = 100 - (100 / (1 + rs))
df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()



# Calculate target variables
target_amount, target_category = calculate_target_variable(df, barsupfront=2, SLTPRatio=1.0, n1=2, n2=2)

# Add target variables to DataFrame
df['target_amount'] = target_amount
df['target_category'] = target_category

# FIXED: Proper win/loss counting logic
wins = 0
losses = 0
both_hit = 0

for i, signal_val in enumerate(df['signal']):
    if signal_val != 0 and target_category[i] != 0:
        if signal_val == 1:  # Bearish signal
            if target_category[i] == 1:  # Hit TP (win for bearish)
                wins += 1
            elif target_category[i] == 2:  # Hit SL (loss for bearish)
                losses += 1
            elif target_category[i] == 3:  # Both hit
                both_hit += 1
        elif signal_val == 2:  # Bullish signal
            if target_category[i] == 2:  # Hit TP (win for bullish)
                wins += 1
            elif target_category[i] == 1:  # Hit SL (loss for bullish)
                losses += 1
            elif target_category[i] == 3:  # Both hit
                both_hit += 1

print(f"Win rate: {wins/(wins+losses)*100:.1f}%" if wins+losses > 0 else "No completed trades")

model = joblib.load('model.pkl')

predictions = []
prediction_markers = [np.nan] * len(df)
#df.to_csv("data/my_trading_signals_apple.csv", index=True)

feature_cols = ['open', 'high', 'low', 'close', 'engulfing', 'star', 'rsi', 'ema_20']
X_all = df[feature_cols]
predictions = model.predict(X_all)

for i in range(len(df)):
    if i < len(predictions):
        prediction = predictions[i]
        if prediction == 1:  # Win for bearish / Loss for bullish
            prediction_markers[i] = df['high'].iloc[i] * 1.001  # Slightly above high
        elif prediction == 2:  # Win for bullish / Loss for bearish  
            prediction_markers[i] = df['low'].iloc[i] * 0.999   # Slightly below low


df.index = pd.to_datetime(df.index)
df = df.sort_index()
print(df.head(5))

apds = [
    mpf.make_addplot(bull_marker, type='scatter', marker='^', color='green', markersize=20),
    mpf.make_addplot(bear_marker, type='scatter', marker='v', color='red', markersize=20),
    mpf.make_addplot(prediction_markers, type='scatter', marker='o', color='blue', markersize=15, alpha=0.7),
]

mpf.plot(
    df,
    type="candle",
    hlines=dict(hlines=hlines, colors=hl_colors, linewidths=0.1),
    style="charles",
    addplot = apds,
    title="Trading Signals with ML Predictions\nGreen: Bullish, Red: Bearish, Blue: ML Prediction"
)
