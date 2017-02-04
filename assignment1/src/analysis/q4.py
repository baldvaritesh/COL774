import numpy as np
from plot import *
from src.analysis.reading import Reader
from src.models.GaussianDiscriminant import GaussianDiscriminant

[(X, restoreData),y] = Reader('q4', [float, str], [1,0])

model = GaussianDiscriminant(sameCov=True)
model.Fit(X, y)
Q4(X,y,'b')
