import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("daily-minimum-temperatures.csv",
                 sep=",", encoding="utf-8",
                 parse_dates=['Date'],
                 index_col='Date')




df.columns = ['temp']

fig,axs = plt.subplots(3)
axs[0].plot(df.index, df.temp, color = 'b')

rolling_mean_bad = df.temp.rolling(window=80).mean()
rolling_mean_nice = df.temp.rolling(window=31).mean()
axs[1].plot(df.index, rolling_mean_bad, color ='r')
axs[2].plot(df.index, rolling_mean_nice, color ='y')
fig.set_size_inches(18.5, 20.5)
fig.show()
