from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

rv_uniform = stats.uniform(0, 1)
rv_normal = stats.norm(10, 10)

rv_uniform_values = rv_uniform.rvs(100)
rv_normal_values = rv_normal.rvs(100)
result = 50 * rv_uniform_values * rv_normal_values + 20

result_mu = 50 * 10 * 0.5 + 20
result_sigma = (50 * 50) * (10 * (1 / 12) + (10 ** 2) * (1 / 12) + (0.5 ** 2) * 10)

fig, ax = plt.subplots(1)
ax.hist(result, 15)
fig.show()
print(result_mu, result_sigma)
fig.clf()

fig, ax = plt.subplots(1)
result_rv = stats.norm(result_mu, result_sigma ** 0.5)
ax.hist(result_rv.rvs(100), 15)
X = np.linspace(-result_sigma ** 0.6, result_sigma, 100)
fig.show()
fig.clf()

fig, ax = plt.subplots(1)
X = np.linspace(-result_sigma ** 0.6, result_sigma ** 0.7, 100)
ax.plot(X, result_rv.pdf(X))
plt.show()

# stats.norm.
