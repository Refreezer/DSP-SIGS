import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.preprocessing import normalize

# def normalize(data_frame):
#     mean = {header: data_frame[header].mean() for header in data_frame}
#     for header in data_frame:
#         s = data_frame[header].std()
#         data_frame[header] -= mean[header]
#         data_frame[header] /= s


df = pd.read_csv("wdbc.data", sep=",", encoding="utf-8", header=None)
bing, ming = df.groupby(df[1])

ming_id = ming[1].index  # [0]
bing_id = bing[1].index  # [0]
print(ming_id)

df_relevant = df
df_relevant.drop([0, 1], axis='columns', inplace=True)

pca = decomposition.PCA(n_components=2)
df_transformed = pca.fit_transform(normalize(df_relevant))
# df_transformed = pca.fit_transform(df_relevant)
print(df_relevant)
print(df_transformed)
print(pca.components_)
df_transformed = pd.DataFrame(data=df_transformed, columns=['x1', 'x2'])
fig, ax = plt.subplots(1)

ming = df_transformed.iloc[list(ming_id)]
bing = df_transformed.iloc[list(bing_id)]

ax.scatter(ming['x1'], ming['x2'], color='red')
ax.scatter(bing['x1'], bing['x2'], color='green')

ax.set_facecolor([0.207, 0.205, 0.204])
# plt.show()
ax.set_title("normalized data.png")
plt.savefig("normalized data.png")

# ax.set_title("unnormalized data.png")
# plt.savefig("unnormalized data.png")

# ming.drop([0, 1], axis='columns', inplace=True)
# bing.drop([0, 1], axis='columns', inplace=True)
# print(bing, ming)
