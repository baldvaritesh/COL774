'''
If you run any module from the interpreter be sure to add the following
two lines to allow relative imports
'''
import sys
if("/home/ritesh/Desktop/IPython/ml/assignment1" not in sys.path):
  sys.path.insert(0, "/home/ritesh/Desktop/IPython/ml/assignment1")

import numpy as np
from src.analysis.plot import *
from src.analysis.reading import Reader

[(X, restoreData), y] = Reader('q2', [float, float], [0, 0])

'''
After viewing the distribution of the data, I chose the following intervals
[-2.0, -1.5], [-1.5, 0.0], [0.0, 1.0], [1.0, 1.5] 
'''
Intervals = np.array([[-2.0, -1.5], [-1.5, 0.0], [0.0, 1.0], [1.0, 1.5]])
model = WeightedLinearRegression(intervals=Intervals)
answer = model.Fit(X, y)
