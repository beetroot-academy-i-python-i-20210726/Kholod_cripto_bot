import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# визначення нахилу лінії ціни
def indSlope(series, n):
    array_sl = [j * 0 for j in range(n - 1)]

    for j in range(n, len(series) + 1):
        y = series[j - n:j]
        x = np.array(range(n))
        x_sc = (x - x.min()) / (x.max() - x.min())
        y_sc = (y - y.min()) / (y.max() - y.min())
        x_sc = sm.add_constant(x_sc)
        model = sm.OLS(y_sc, x_sc)
        results = model.fit()
        array_sl.append(results.params[-1])
    slope_angle = (np.rad2deg(np.arctan(np.array(array_sl))))
    return np.array(slope_angle)

# Індикатор ATR
def indATR(source_DF, n):
    df = source_DF.copy()
    df['H-L'] = abs(df['high'] - df['low'])
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1, skipna=False)
    df['ATR'] = df['TR'].rolling(n).mean()
    df_temp = df.drop(['H-L', 'H-PC', 'L-PC'], axis=1)
    return df_temp

# створити фрейм даних

def PrepareDF(DF):
    ohlc = DF.iloc[:, [0, 1, 2, 3, 4, 5]]
    ohlc.columns = ["date", "open", "high", "low", "close", "volume"]
    ohlc = ohlc.set_index('date')
    df = indATR(ohlc, 14).reset_index()
    df['slope'] = indSlope(df['close'], 5)
    df['channel_max'] = df['high'].rolling(10).max()
    df['channel_min'] = df['low'].rolling(10).min()
    df['position_in_channel'] = (df['close'] - df['channel_min']) / (df['channel_max'] - df['channel_min'])
    df = df.set_index('date')
    df = df.reset_index()
    return (df)

# знайти локальний мінімум/локальний максимум
def isLCC(DF, i):
    df = DF.copy()
    LCC = 0

    if df['close'][i] <= df['close'][i + 1] and df['close'][i] <= df['close'][i - 1] and df['close'][i + 1] > \
            df['close'][i - 1]:
        # локальний мінімум
        LCC = i - 1;
    return LCC


def isHCC(DF, i):
    df = DF.copy()
    HCC = 0
    if df['close'][i] >= df['close'][i + 1] and df['close'][i] >= df['close'][i - 1] and df['close'][i + 1] < \
            df['close'][i - 1]:
        # локальний максимум
        HCC = i;
    return HCC

def getMaxMinChannel(DF, n):
    maxx = 0
    minn = DF['low'].max()
    for i in range(1, n):
        if maxx < DF['high'][len(DF) - i]:
            maxx = DF['high'][len(DF) - i]
        if minn > DF['low'][len(DF) - i]:
            minn = DF['low'][len(DF) - i]
    return (maxx, minn)

apiKey = 'BK53YCTI6Q5VENNY'

interval_var = '5min'
symbol = 'ETH'

path = 'https://www.alphavantage.co/query?function=CRYPTO_INTRADAY&symbol=' + symbol + '&market=USD&interval='\
       + interval_var + '&apikey=' + apiKey + '&datatype=csv&outputsize=full'
df = pd.read_csv(path)

# Реверс даних
df = df[::-1]

prepared_df = PrepareDF(df)

lend = len(prepared_df)

prepared_df['hcc'] = [None] * lend
prepared_df['lcc'] = [None] * lend

prepared_df[10:25]['close'].plot()

for i in range(4, lend - 1):
    if isHCC(prepared_df, i) > 0:
        prepared_df.at[i, 'hcc'] = prepared_df['close'][i]
    if isLCC(prepared_df, i) > 0:
        prepared_df.at[i, 'lcc'] = prepared_df['close'][i]

# Графічне відображення

aa = prepared_df[400:600]
aa = aa.reset_index()

labels = ['close', "hcc", "lcc", "channel_max", "channel_min"]
labels_line = ['--', "*-", "*-", "g-", "r-"]

j = 0
x = pd.DataFrame()
y = pd.DataFrame()
for i in labels:
    x[j] = aa['index']
    y[j] = aa[i]
    j = j + 1

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

fig.suptitle('Deals')
fig.set_size_inches(20, 10)

for j in range(0, len(labels)):
    ax1.plot(x[j], y[j], labels_line[j])

ax1.set_ylabel('Price')
ax1.grid(True)

ax2.plot(x[0], aa['slope'], '.-')  # EMA
ax3.plot(x[0], aa['position_in_channel'], '.-')  # EMA

ax2.grid(True)
ax3.grid(True)

plt.show()