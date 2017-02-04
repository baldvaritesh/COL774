import numpy as np
from src.analysis.plot import *
from src.analysis.reading import Reader
from src.models.LogisticRegression import LogisticRegression

[(X, restoreData), y] = Reader('q3', [float, float], [1, 0])

model = LogisticRegression()
answer = model.Fit(X, y)

Q3(X, y, model.Thetas)
