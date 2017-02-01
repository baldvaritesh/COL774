'''
If you run any module from the interpreter be sure to add the following
two lines to allow relative imports
'''
import sys
if("/home/ritesh/Desktop/IPython/ml/assignment1" not in sys.path):
  sys.path.insert(0, "/home/ritesh/Desktop/IPython/ml/assignment1")

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