import numpy as np
import matplotlib.pyplot as plt

alc = [[], [], []]
acid = [[], [], []]
colors = ['r', 'deeppink', 'c']

with open("wine.data") as file:
    for line in file.readlines():
        values = line.split(",")[:3]
        type = (int(values[0]) - 1)
        alc[type].append(float(values[1]))
        acid[type].append(float(values[2]))

fig, ax = plt.subplots()
print(acid)
ax.set_title("no standartization")
ax.set_facecolor([0.207, 0.205, 0.204])
for i in range(3):
    ax.scatter(alc[i], acid[i], c=colors[i])

fig.savefig("no_standartization.png")
old_acid = acid
old_alc = alc
plt.clf()

hypercube = {'alc': (min(alc[0] + alc[1] + alc[2]), max(alc[0] + alc[1] + alc[2])),
             'acid': (min(acid[0] + acid[1] + acid[2]), max(acid[0] + acid[1] + acid[2]))}

fig, ax = plt.subplots()

ax.set_title("hyper cube standartization")
ax.set_facecolor([0.207, 0.205, 0.204])

alc = [[(alc_vec_el - hypercube['alc'][0]) / (hypercube['alc'][1] - hypercube['alc'][0]) for alc_vec_el in alc_vec]
       for alc_vec in alc]

acid = [
    [(acid_vec_el - hypercube['acid'][0]) / (hypercube['acid'][1] - hypercube['acid'][0]) for acid_vec_el in acid_vec]
    for acid_vec in acid]

print(list(zip(acid, alc)))

for i in range(3):
    ax.scatter(alc[i], acid[i], c=colors[i])
fig.savefig("hyper_cube_standartization.png")
plt.clf()
fig, ax = plt.subplots()
ax.set_title("statistics standartization")
ax.set_facecolor([0.207, 0.205, 0.204])

alc = old_alc
acid = old_acid
average_alc = sum(alc[0] + alc[1] + alc[2]) / len(alc[0] + alc[1] + alc[2])
average_acid = sum(acid[0] + acid[1] + acid[2]) / len(acid[0] + acid[1] + acid[2])
s_alc = sum(list(map(lambda el: el - average_alc, alc[0] + alc[1] + alc[2]))) / (len(alc[0]) + len(alc[1]) + len(alc[2]))
s_acid = sum(list(map(lambda el: el - average_acid, acid[0] + acid[1] + acid[2]))) / (len(acid[0]) + len(acid[1]) + len(acid[2]))
print(sum(list(map(lambda el: el - average_alc, alc[0] + alc[1] + alc[2]))))
print(average_alc, len(alc[0]) + len(alc[1]) + len(alc[2]),s_alc)

alc = [[(alc_vec_el - average_alc) / s_alc for alc_vec_el in alc_vec]
       for alc_vec in alc]
acid = [[(acid_vec_el - average_acid) / s_acid for acid_vec_el in acid_vec]
        for acid_vec in acid]

for i in range(3):
    ax.scatter(alc[i], acid[i], c=colors[i])
fig.savefig("statistics_standartization.png")
