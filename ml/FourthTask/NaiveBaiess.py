from __future__ import division
from collections import defaultdict
from math import log
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import plot_confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.preprocessing import normalize
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

train_df = pd.read_csv("train.csv", sep=",", encoding="utf-8", header=None)
X = np.array(train_df.drop([0], axis='columns'))
train_y = np.array(train_df[0])
nb = GaussianNB()
nb.fit(X, train_y)


test_df = pd.read_csv("test.csv", sep=",", encoding="utf-8", header=None)
test_x = np.array(test_df.drop([0], axis='columns'))
actualY = np.array(test_df[0])
plot_confusion_matrix(nb, test_x, actualY)
plt.savefig("orig.png")
print(accuracy_score(actualY, nb.predict(test_x), normalize=False))


# PCA
pca = decomposition.PCA(n_components=37)
# TRAIN
train_x = pca.fit_transform(normalize(X.copy(), axis=1, norm='l2'))
nb = GaussianNB()

# TEST
x = pca.transform(normalize(test_x.copy(), axis=1, norm='l2'))
nb.fit(train_x, train_y)
plot_confusion_matrix(nb, x, actualY)
plt.savefig("with_pca.png")
print(accuracy_score(actualY, nb.predict(x), normalize=False))


# plt.show()
# plt.clf()

# cnt = 0
# i = 0
#
# # for predicted, actual in zip(predictedY, actualY):
# #     if predicted != actual:
# #         cnt += 1
# #         print(str(i) + ") " + predicted + " != " + actual)
# #     i += 1
# #
# # print(cnt)
