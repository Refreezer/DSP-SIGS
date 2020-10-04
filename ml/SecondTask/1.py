import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition


def center_data_frame(data_frame):
    mean = {header: data_frame[header].mean() for header in data_frame}
    for header in data_frame:
        data_frame[header] -= mean[header]

# 1
df = data = pd.read_csv("pca.csv", sep=",", encoding="utf-8", header=None)
for i in df:
    print(i)
df.columns = ['x1', 'x2']
fig, ax = plt.subplots(1)
ax.set_facecolor([0.207, 0.205, 0.204])
ax.scatter(df['x1'], df['x2'])
ax.set_title('original data')
plt.savefig("OriginalData.png")
plt.clf()

# 2
df2comp = df
center_data_frame(df2comp)
pca = decomposition.PCA(n_components=2)
pca.fit(df2comp)
df2_pca = pca.transform(df2comp)
df2_pca = pd.DataFrame(data=df2_pca, columns=['x1', 'x2'])
fig, ax = plt.subplots(1)
ax.set_facecolor([0.207, 0.205, 0.204])
ax.set_title('pca2 & original axes')
for i, component in enumerate(pca.components_):
    ax.arrow(0, 0, component[0], component[1], width=0.01, head_length=0.04, color='purple')

ax.scatter([i for i in df2_pca['x1']], [i for i in df2_pca['x2']])
print(pca.components_)
plt.plot()
plt.savefig("pca2.png")
plt.clf()

# 3
df1comp = df
center_data_frame(df1comp)
pca = decomposition.PCA(n_components=1)
pca.fit(df1comp)
data_vec = pca.transform(df1comp)
# df.plot(kind='scatter', x='x1', y='x2')
df1_pca = pd.DataFrame(data=data_vec, columns=['x1'])

df1_pca['x1'] = data_vec * np.abs(np.linalg.norm(pca.components_[0][0]))
df1_pca['x2'] = data_vec * np.abs(np.linalg.norm(pca.components_[0][1]))
# df1_pca.plot(kind='scatter', x='x1', y='x2', color='r')
combined_df = pd.concat([df, df1_pca])
fig, ax = plt.subplots(1)
ax.set_facecolor([0.207, 0.205, 0.204])
ax.set_title('pca1 & original')
ax.scatter(combined_df['x1'][:len(df)], combined_df['x2'][:len(df)], color='b')
ax.scatter(combined_df['x1'][len(df):], combined_df['x2'][len(df):], color='r')
plt.savefig("pca1.png")
