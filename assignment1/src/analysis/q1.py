import numpy as np
from plot import *
from src.analysis.reading import Reader
from src.models.LinearRegression import LinearRegression

[(X, restoreData),y] = Reader('q1', [float, float], [0,0])

model = LinearRegression(count=1, storeCostFunction=True, eta=1.3)
answer = np.array(model.Fit(X, y))

Q1d(X, y, answer, 'e4')
