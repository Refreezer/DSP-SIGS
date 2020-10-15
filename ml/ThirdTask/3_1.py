import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

df = pd.read_csv("t.csv", sep=",", encoding="utf-8", index_col=0)

x = df['total']
y = np.array(list(df['t']))
x = np.reshape(list(x), (-1, 1))

# lr = linear_model.LinearRegression()
# lr.fit(x, y)
#
ransac = linear_model.RANSACRegressor()
ransac.fit(x, y)
#
inlier_mask = ransac.inlier_mask_

outlier_mask = np.logical_not(inlier_mask)
inlier_mask = [i for i, boo in enumerate(inlier_mask) if boo == True]
outlier_mask = [i for i, boo in enumerate(outlier_mask) if boo == True]

fig, ax = plt.subplots(1)
ax.set_facecolor([0.207, 0.205, 0.204])
x_line = x
# y_line = lr.predict(x_line.reshape(-1,1))
line_y_ransac = ransac.predict(x_line.reshape(-1,1))
#
lw = 2
plt.scatter(x[inlier_mask], y[inlier_mask], color='cyan', marker='.',
            label='Inliers')
plt.scatter(x[outlier_mask], y[outlier_mask], color='red', marker='.',
            label='Noize')
# plt.plot(x_line, y_line, color='navy', linewidth=lw, label='Linear regressor')
plt.plot(x_line, line_y_ransac, color='cornflowerblue', linewidth=lw,
         label='RANSAC regressor')
plt.legend(loc='lower right')
plt.xlabel("Input")
plt.ylabel("Response")
plt.show()






