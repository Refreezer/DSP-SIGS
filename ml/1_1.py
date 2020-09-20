import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("daily-minimum-temperatures.csv",
                 sep=",", encoding="utf-8",
                 parse_dates=['Date'],
                 index_col='Date')




df.columns = ['temp']

fig,axs = plt.subplots(3)
axs[0].plot(df.index, df.temp, color = 'b')

rolling_mean = df.temp.rolling(window=20).mean()
rolling_mean2 = df.temp.rolling(window=50).mean()
axs[1].plot(df.index, rolling_mean, color ='r')
axs[2].plot(df.index, rolling_mean2, color ='y')
fig.show()
