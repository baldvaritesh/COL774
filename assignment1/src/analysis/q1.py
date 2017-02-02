import numpy as np
from plot import *
from src.analysis.reading import Reader
from src.models.LinearRegression import LinearRegression

[(X, restoreData),y] = Reader('q1', [float, float], [0,0])

'''
Do different parts by setting the corresponding arguments
'''
model = LinearRegression()
answer = model.Fit(X, y)

Q1b(X, y, model.Thetas)
Q1c(X, y, answer)
