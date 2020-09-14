import numpy as np

data = []

with open("wine.data") as file:
    for line in file.readlines():
        data += [line.split(",")[1:3]]

print(data)