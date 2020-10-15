import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from scipy import stats


def swap_each_third(vec, val):
    for i in range(2, len(vec), 3):
        vec[i] = val


rv_normal = stats.norm(0, 1)
x_train = np.array(rv_normal.rvs(400))
x_test = np.array(rv_normal.rvs(200))

y_train = np.sin(x_train)
y_test = np.sin(x_train)

x3_train, x10_train, y3_train, y10_train = x_train.copy(), x_train.copy(), y_train.copy(), y_train.copy()

swap_each_third(x3_train, 3.0)
swap_each_third(y3_train, 3.0)
swap_each_third(x10_train, 10.0)
swap_each_third(y10_train, 10.0)


def make_regressions(x, y, x_test):
    x = x.reshape(-1, 1)
    lr = linear_model.LinearRegression()
    lr.fit(x, y)

    ransac = linear_model.RANSACRegressor()
    ransac.fit(x, y)
    r_inlier_mask = ransac.inlier_mask_
    r_outlier_mask = np.logical_not(r_inlier_mask)

    hueber = linear_model.HuberRegressor()
    hueber.fit(x, y)
    h_outlier_mask = hueber.outliers_
    h_inlier_mask = np.logical_not(h_outlier_mask)

    x_line = x_test.reshape(-1, 1)
    lr_line = lr.predict(x_line)
    ransac_line = ransac.predict(x_line)
    hueber_line = hueber.predict(x_line)

    fig, ax = plt.subplots(1)
    ax.scatter(x_test, np.sin(x_test), color='grey', marker='.')
    # ax.scatter(x_test[range(2, len(x_test), 3)], np.sin(x_test[range(2, len(x_test), 3)]), color='red', marker='.')
    # ax.scatter(x, y, color='grey', marker='.')
    # ax.scatter(x[range(2, len(x_test), 3)], y[range(2, len(x_test), 3)], color='r', marker='.')
    ax.plot(x_line, lr_line, color='b')
    ax.plot(x_line, hueber_line, color='red', linewidth = 2)
    ax.plot(x_line, ransac_line, color='cyan')

    # fig, ax = plt.subplots(3)
    # ax[0].scatter(x_test, np.sin(x_test), color='grey')
    # ax[0].plot(x_line, lr_line, color='red')
    #
    # ax[1].scatter(x_test[h_inlier_mask], np.sin(x_test[h_inlier_mask]), marker='.', label='Inliers', color='purple')
    # ax[1].scatter(x_test[h_outlier_mask], np.sin(x_test[h_outlier_mask]), marker='.', label='Outliers', color='grey')
    # ax[1].plot(x_line, hueber_line, color='purple')
    #
    # ax[2].scatter(x_test[r_inlier_mask], np.sin(x_test[r_inlier_mask]), marker='.', label='Inliers', color='purple')
    # ax[2].scatter(x_test[r_outlier_mask], np.sin(x_test[r_outlier_mask]), marker='.', label='Outliers', color='cyan')
    # ax[2].plot(x_line, ransac_line, color='cyan')
    plt.show()
    plt.clf()


make_regressions(x3_train, y_train, x_test)
make_regressions(x_train, y3_train, x_test)
make_regressions(x10_train, y_train, x_test)
make_regressions(x_train, y10_train, x_test)

