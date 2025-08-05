# %% CELL 1 import numpy as np
import pandas as pd
import mplfinance as mpf
from dotenv import load_dotenv
from twelvedata import TDClient
import os

#df = pd.read_csv("gold_4mo_daily.csv")
#df['datetime'] = pd.to_datetime(df['datetime'])
#df.set_index('datetime', inplace=True)
#df = df[-200:]

load_dotenv("secrets.env")

api_key = os.getenv("TD_API_KEY")

td = TDClient(apikey=api_key)
ts = td.time_series(
    symbol="XAU/USD",
    interval="4h",
    #interval="15min",
    outputsize=500,
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
n1 = 2
n2 = 2
threshold = 0.006
support, resistance = calculateSR(df,n1,n2,threshold)
hlines = support + resistance
hl_colors = ['green'] * len(support) + ['red'] * len(resistance)

# %% CELL 6
mpf.plot(
    df,
    type="candle",
    hlines=dict(hlines=hlines, colors=hl_colors, linewidths=0.5),
    style="binance"
)
