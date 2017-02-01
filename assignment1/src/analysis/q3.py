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
from src.models.LogisticRegression import LogisticRegression

[(X, restoreData), y] = Reader('q3', [float, float], [1, 0])

model = LogisticRegression()
answer = model.Fit(X, y)

Q3b(X, y, model.Thetas)