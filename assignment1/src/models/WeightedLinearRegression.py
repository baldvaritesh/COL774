import numpy as np
from src.models.common import *

class WeightedLinearRegression():
  '''
  Creates a WeightedLinearRegression object
  '''

  def __init__(self, weighted = False, bandwidth = 1.0, intervals = []):
    '''
    Similar like simple regression
    1. Parameter for bandwidth added
    2. Parameter for intervals on which the model would be trained, dimension: R^(n+1)
    '''
    self.Thetas = []
    self.Bandwidth = bandwidth
    self.returnValue = None
    self.Intervals = intervals
    self.Weighted = weighted

  def __ChoosePoint(self, X, interval):
    '''
    Choose the middle-most point in the interval for training
    1. Will cross product be more efficient for checking whether point lies inside??
    2. Point chosen is the closest to the centroid of the nd cuboid
    '''
    intervalPoints = []
    for i in range(0, self.M):
      for j in range(0, len(interval)):
        if((X[i][j+1] >= interval[j][0]) and (X[i][j+1] < interval[j][1])):
          intervalPoints.append(X[i])

    centroid = np.zeros(shape=(len(interval)))
    for i in range(0, len(interval)):
      centroid[i] = ( interval[i][0] + interval[i][1] )/ 2

    minPoint = None
    minNorm = np.inf
    for point in intervalPoints:
      currNorm = np.linalg.norm(point - centroid)
      if(currNorm < minNorm):
        minNorm = currNorm
        minPoint = point
    return minPoint

  def Fit(self, X, y):

    def _WMatrix(self, point):
      nonlocal X
      return np.diag([np.exp(- np.dot((point - x).T, point - x) / (2 * (self.Bandwidth ** 2)) ) for x in X])

    def _Update(self, idx = None, _W = None ):
      '''
      idx and W are needed if weighted regression occurs
      '''
      nonlocal X, y

      if(self.Weighted == True):
        firstTerm = np.dot(X.T, np.dot(W, X))
        InvTerm = np.linalg.inv(firstTerm)
        return np.dot(InvTerm, np.dot(X.T, np.dot(W, y)))

      else:
        firstTerm = np.dot(X.T, X)
        InvTerm = np.linalg.inv(firstTerm)
        return np.dot(InvTerm, np.dot(X.T, y))

    (X,y) = PreProcessFit(X, y, model='regression')
    self.M = len(X)
    self.N = len(X.T)
    '''
    Parameters for the model
    Number of parameters thus depend on the number of intervals
    '''
    if(self.Weighted == True):
      self.Thetas = np.zeros(shape=(len(self.Intervals), self.N))
      self.Points = []
      for interval in self.Intervals:
        self.Points.append(self.__ChoosePoint(X, interval))
      self.Points = np.array(self.Points)

      for i in range(0, len(self.Intervals)):
        W = _WMatrix(self, self.Points[i])
        self.Thetas[i] = np.squeeze((_Update(self, i, W)))
    else:
      self.Thetas = np.zeros(shape=(self.N, 1))
      self.Thetas = _Update(self)

    return self.returnValue
