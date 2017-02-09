import numpy as np
from src.analysis.plot import *
from src.analysis.reading import Reader
from src.models.WeightedLinearRegression import WeightedLinearRegression

[(X, restoreData), y] = Reader('q2', [float, float], [0, 0])

model = WeightedLinearRegression(weighted = True, bandwidth = 0.025)
answer = model.Fit(X, y)
Q2(X, y, 'c5', answer)
