import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform, pdist
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

iris_df = pd.read_csv("iris.data", sep=",", encoding="utf-8", header=None)

relevant_df = pd.DataFrame(data=iris_df, columns=[0, 1, 2, 3, 4])
relevant_df.drop([4], axis='columns', inplace=True)

Iris_setosa, Iris_versicolor, Iris_virginica = map(lambda el: el[1].drop([4], axis='columns'),
                                                   iris_df.groupby(iris_df[4]))
# .drop([4], axis='columns')

Iris_setosa_id, Iris_versicolor_id, Iris_virginica_id = Iris_setosa.index, Iris_versicolor.index, Iris_virginica.index

emb = MDS(n_components=2)

transformed = emb.fit_transform(relevant_df)
fig, ax = plt.subplots(1)
ax.scatter([i[0] for i in transformed[Iris_setosa_id]],[i[1] for i in transformed[Iris_setosa_id]], color='c')
ax.scatter([i[0] for i in transformed[Iris_versicolor_id]],[i[1] for i in transformed[Iris_versicolor_id]], color='r')
ax.scatter([i[0] for i in transformed[Iris_virginica_id]],[i[1] for i in transformed[Iris_virginica_id]], color='g')

plt.show()
print(np.shape(transformed))




distances = []
transformed_distances = []

for i in range(len(iris_df) - 1):
    for j in range(i + 1, len(iris_df)):
        distances.append(np.linalg.norm((relevant_df.iloc[i]) - relevant_df.iloc[j]))
        transformed_distances.append(np.linalg.norm(transformed[i] - transformed[j]))

fig, ax = plt.subplots(1)
ax.scatter(distances, transformed_distances, s = 1)
plt.show()
