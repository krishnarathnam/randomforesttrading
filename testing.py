

def calculate_target_variable(df, barsupfront=3, SLTPRatio=2.0, n1=2, n2=2):
    """
    Calculate target variable for ML training based on your technical analysis signals
    Returns profit/loss amounts for each signal
    """
    length = len(df)
    high = list(df['high'])
    low = list(df['low'])
    close = list(df['close'])
    open = list(df['open'])
    signal = list(df['signal'])
    trendcat = [0] * length
    amount = [0] * length
    
    SL = 0
    TP = 0
    
    for line in range(n1, length-barsupfront-n2):
        if signal[line] == 1:  # Bearish signal (expecting price to go down)
            SL = max(high[line-1:line+1])  # Stop Loss = highest of current + previous bar
            TP = close[line] - SLTPRatio * (SL - close[line])  # Take Profit below entry
            
            for i in range(1, barsupfront+1):
                if line+i < length:  # Check bounds
                    if low[line+i] <= TP and high[line+i] >= SL:
                        trendcat[line] = 3  # Both TP and SL hit
                        break
                    elif low[line+i] <= TP:
                        trendcat[line] = 1  # Win - hit take profit
                        amount[line] = close[line] - low[line+i]
                        break
                    elif high[line+i] >= SL:
                        trendcat[line] = 2  # Loss - hit stop loss
                        amount[line] = close[line] - high[line+i]
                        break

        elif signal[line] == 2:  # Bullish signal (expecting price to go up)
            SL = min(low[line-1:line+1])  # Stop Loss = lowest of current + previous bar
            TP = close[line] + SLTPRatio * (close[line] - SL)  # Take Profit above entry
            
            for i in range(1, barsupfront+1):
                if line+i < length:  # Check bounds
                    if high[line+i] >= TP and low[line+i] <= SL:
                        trendcat[line] = 3  # Both TP and SL hit
                        break
                    elif high[line+i] >= TP:
                        trendcat[line] = 2  # Win - hit take profit
                        amount[line] = high[line+i] - close[line]
                        break
                    elif low[line+i] <= SL:
                        trendcat[line] = 1  # Loss - hit stop loss
                        amount[line] = low[line+i] - close[line]
                        break
    
    return amount, trendcat
