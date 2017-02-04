import numpy as np
from src.analysis.plot import *
from src.analysis.reading import Reader
from src.models.WeightedLinearRegression import WeightedLinearRegression

[(X, restoreData), y] = Reader('q2', [float, float], [0, 0])

Intervals = np.array([[[-2.0, -1.5]], [[-1.5, -1.0]], [[-1.0, -0.5]], [[-0.5, 0.0]], [[0.0, 0.5]], [[0.5, 1.0]], [[1.0, 1.7]]])
model = WeightedLinearRegression(weighted = True, bandwidth = 0.15, intervals = Intervals)
answer = model.Fit(X, y)
Q2(X, y, 'b', model.Thetas, Intervals)
