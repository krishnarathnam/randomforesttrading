# %% CELL 1 import numpy as np
import numpy as np
import pandas as pd
import mplfinance as mpf
from dotenv import load_dotenv
from twelvedata import TDClient
import os
import yfinance as yf

#df = yf.download("SWIGGY.NS", period="1y")
#df.reset_index(inplace=True)
#df = df.iloc[2:] 
#df['Date'] = pd.to_datetime(df['Date'])
#df.columns = ['date', 'close', 'high', 'low', 'open', 'volume']
#df.set_index('date', inplace=True)



load_dotenv("secrets.env")

api_key = os.getenv("TD_API_KEY")

td = TDClient(apikey=api_key)
ts = td.time_series(
    symbol="XAU/USD",
    interval="4h",
    outputsize=200,
    #interval="15min",
    #outputsize=500,
    timezone="UTC"
)

df = ts.as_pandas()
df = df.sort_index()

# %% CELL 2
print(df.head())

# %% CELL 3
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


# %% CELL 4
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



# %% CELL 4
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
        
        #print(df.RSI[row])
        #  |
        # _|_
        #|__|
        # |
        # 
        
        if (ratio1[row]>2.5 and lowdiff[row]<0.3*highdiff[row] and bodydiff[row]>0.03 ): #and df.RSI[row]>60 and df.RSI[row]<80 
            signal[row] = 1
        
        #elif (ratio2[row-1]>2.5 and highdiff[row-1]<0.23*lowdiff[row-1] and bodydiff[row-1]>0.03 and bodydiff[row]>0.04 and close[row]>open[row] and close[row]>high[row-1] and df.RSI[row]<55 and df.RSI[row]>30):
        #    signal[row] = 2
        # _|_
        #|__|
        # |
        # |
        
        elif (ratio2[row]>2.5 and highdiff[row]<0.23*lowdiff[row] and bodydiff[row]>0.03 ):# and df.RSI[row]<55 and df.RSI[row]>10
            signal[row]=2
    return signal


# %% CELL 5
def closeResistance(df,l,levels,lim):
    if len(levels)==0:
        return 0
    nearest_resistance = min(levels, key=lambda x: abs(x - df['high'].iloc[l]))

#    c1 = abs(df['high'].iloc[l] - nearest_resistance) <= lim
#    c2 = abs(max(df['open'].iloc[l], df['close'].iloc[l]) - nearest_resistance) <= lim
#    c3 = min(df['open'].iloc[l], df['close'].iloc[l]) < nearest_resistance
#    c4 = df['low'].iloc[l] < nearest_resistance
#    if( (c1 or c2) and c3 and c4 ):
#        return 1
#    else:
#        return 0
    
    c1 = df['high'].iloc[l] >= nearest_resistance  # Wick touches or goes above resistance
    c2 = abs(df['high'].iloc[l] - nearest_resistance) <= lim and df['high'].iloc[l] < nearest_resistance
    c3 = max(df['open'].iloc[l], df['close'].iloc[l]) < nearest_resistance  # Body stays below
    if (c1 or c2) and c3:
        return 1
    else:
        return 0
    
def closeSupport(df,l,levels,lim):
    if len(levels)==0:
        return 0
    nearest_support = min(levels, key=lambda x: abs(x - df['low'].iloc[l]))

    #c1 = abs(df['low'].iloc[l] - nearest_support) <= lim
    #c2 = abs(min(df['open'].iloc[l], df['close'].iloc[l]) - nearest_support) <= lim
    #c3 = max(df['open'].iloc[l], df['close'].iloc[l]) > nearest_support
    #c4 = df['high'].iloc[l] > nearest_support
    #if( (c1 or c2) and c3 and c4 ):
    #    return 1
    #else:
    #    return 0
    c1 = df['low'].iloc[l] <= nearest_support
    c2 = abs(df['low'].iloc[l] - nearest_support) <= lim and df['low'].iloc[l] > nearest_support
    c3 = max(df['open'].iloc[l], df['close'].iloc[l]) > nearest_support  # Body stays below
    if (c1 or c2) and c3:
        return 1
    else:
        return 0

# %% CELL 6
n1 = 2
n2 = 2
threshold = 0.006
df['engulfing'] = Revsignal1(df)
df['star'] = Revsignal2(df)
support, resistance = calculateSR(df,n1,n2,threshold)
hlines = support + resistance
hl_colors = ['green'] * len(support) + ['red'] * len(resistance)

#engulfing = df['engulfing']
#engulfing_bull_marker = [np.nan] * len(df)
#engulfing_bear_marker = [np.nan] * len(df)
#for i in range(len(df)):
#    if engulfing.iloc[i] == 2:
#        engulfing_bull_marker[i] = df['low'].iloc[i] * 1
#    elif engulfing.iloc[i] == 1:
#        engulfing_bear_marker[i] = df['high'].iloc[i] * 1
#
#star = df['star']
#star_bull_marker = [np.nan] * len(df)
#star_bear_marker = [np.nan] * len(df)
#for i in range(len(df)):
#    if star.iloc[i] == 2:
#        star_bull_marker[i] = df['low'].iloc[i] * 1
#    elif star.iloc[i] == 1:
#        star_bear_marker[i] = df['high'].iloc[i] * 1
#

#apds = [
#    mpf.make_addplot(engulfing_bull_marker, type='scatter', marker='^', color='green', markersize=20),
#    mpf.make_addplot(engulfing_bear_marker, type='scatter', marker='v', color='red', markersize=20),
#    mpf.make_addplot(star_bull_marker, type='scatter', marker='o', color='green', markersize=20),
#    mpf.make_addplot(star_bear_marker, type='scatter', marker='o', color='red', markersize=20)
#]

df['signal'] = 0
for row in range(n1,len(df)-n2):
    if((df['engulfing'].iloc[row]==1 or df['star'].iloc[row]==1) and closeResistance(df,row,resistance,0.0015)):
        df.loc[df.index[row], 'signal'] = 1
    elif((df['engulfing'].iloc[row]==2 or df['star'].iloc[row]==2) and closeSupport(df,row,support,0.0015)):
        df.loc[df.index[row], 'signal'] = 2
    else:
        df.loc[df.index[row], 'signal'] = 0

print(df.tail())

signal = df['signal']
bull_marker = [np.nan] * len(df)
bear_marker = [np.nan] * len(df)
for i in range(len(df)):
    if signal.iloc[i] == 2:
        bull_marker[i] = df['low'].iloc[i] * 1
    elif signal.iloc[i] == 1:
        bear_marker[i] = df['high'].iloc[i] * 1

apds = [
    mpf.make_addplot(bull_marker, type='scatter', marker='^', color='green', markersize=20),
    mpf.make_addplot(bear_marker, type='scatter', marker='v', color='red', markersize=20),
]

print((df['signal'] == 1).count())
# %% CELL 6
df.index = pd.to_datetime(df.index)
df = df.sort_index()
mpf.plot(
    df,
    type="candle",
    hlines=dict(hlines=hlines, colors=hl_colors, linewidths=0.1),
    style="charles",
    addplot = apds
)

# %%
