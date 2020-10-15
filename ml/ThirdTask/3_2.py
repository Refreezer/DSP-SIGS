import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets


df = pd.read_csv("t.csv", sep=",", encoding="utf-8", index_col=0)

s = np.array([1 if el == "Yes" else 0 for el in df['s']])

time = np.array([1 if el == "D" else 0 for el in df['time']])

y = np.array(df['t'])
x = np.array([[i, j, k] for i, j, k in zip(df['total'], s, time)])

ransac = linear_model.RANSACRegressor()
ransac.fit(x, y)
inlier_mask = ransac.inlier_mask_

outlier_mask = np.logical_not(inlier_mask)
inlier_mask = [i for i, boo in enumerate(inlier_mask) if boo == True]
outlier_mask = [i for i, boo in enumerate(outlier_mask) if boo == True]

x_line = x
print(np.shape(x_line), np.shape(y))
line_y_ransac = ransac.predict(x_line)

fig, ax = plt.subplots(1)
ax.set_facecolor([0.207, 0.205, 0.204])
lw = 2
# plt.scatter([el[0] for el in x[inlier_mask]], y[inlier_mask], color='yellowgreen', marker='.',
#             label='Inliers')
# plt.scatter([el[0] for el in x[outlier_mask]], y[outlier_mask], color='gold', marker='.',
#             label='Noize')
plt.scatter(df['t'].iloc[inlier_mask], line_y_ransac[inlier_mask], color='cyan')
plt.scatter(df['t'].iloc[outlier_mask], line_y_ransac[outlier_mask], color='red')

plt.legend(loc='lower right')
plt.xlabel("Input")
plt.ylabel("Response")
plt.show()
