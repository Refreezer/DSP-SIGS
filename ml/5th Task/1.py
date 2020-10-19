import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

df = pd.read_csv("wine.data", sep=",", encoding="utf-8")
print (df['1'])