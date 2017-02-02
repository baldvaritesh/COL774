import numpy as np
from src.models.common import *

class WeightedLinearRegression():
  '''
  Creates a WeightedLinearRegression object
  '''

  def __init__(self, bandwidth = 1.0, eta = 0.00001, intervals = []):
    '''
    Similar like simple regression
    1. Parameter for bandwidth added
    2. Parameter for intervals on which the model would be trained, dimension: R^(n+1)
    '''
    self.PreviousThetas = []
    self.Thetas = []
    self.Bandwidth = bandwidth
    self.Eta = eta
    self.returnValue = None
    self.Intervals = intervals

  def __ChoosePoint(self, X, interval):
    '''
    Choose the middle-most point in the interval for training
    1. Will cross product be more efficient for checking whether point lies inside??
    2. Point chosen is the closest to the centroid of the nd cuboid
    '''
    intervalPoints = []
    for i in range(0, self.M):
      for j in range(0, len(interval)):
        if(X[i][j] >= interval[j]['start'] and X[i][j] <= interval[j]['end']):
          intervalPoints.append((X[i], i))

    centroid = np.zeros()



  def Fit(X, y):

    (X,y) = PreProcessFit(X, y)
    self.M = len(X)
    self.N = len(X.T)
    '''
    Parameters for the model
    Number of parameters thus depend on the number of intervals
    '''
    self.Thetas = np.zeros(shape=(len(self.Intervals), self.N))

    return self.returnValue

  def Predict(X):

