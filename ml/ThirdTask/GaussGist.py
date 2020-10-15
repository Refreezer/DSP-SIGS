import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

mu, sigma = 100, 15
rv = stats.norm()
s = rv.rvs(1000)
X = np.linspace(-4, 4, 100)
fig, ax = plt.subplots(1)
ax.set_facecolor([0.207, 0.205, 0.204])
ax.hist(s, 20, color='r')
ax.plot(X, 50 * (max(s) - min (s)) *rv.pdf(X))
plt.show()
