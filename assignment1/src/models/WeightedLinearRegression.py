import numpy as np
from src.models.common import *

class WeightedLinearRegression():
  '''
  Creates a WeightedLinearRegression object
  '''

  def __init__(self, weighted = False, bandwidth = 1.0):
    '''
    Similar like simple regression
    1. Parameter for bandwidth added
    '''
    self.Thetas = []
    self.Bandwidth = bandwidth
    self.returnValue = None
    self.Weighted = weighted

  def Fit(self, X, y):

    def _WMatrix(self, point):
      nonlocal X
      return np.diag([np.exp(- np.dot((point - x).T, point - x) / (2 * (self.Bandwidth ** 2)) ) for x in X])

    def _Update(self, _W = None ):
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
    self.returnValue = []
    '''
    Parameters for the model
    Number of parameters thus depend on the number of intervals
    '''
    if(self.Weighted == True):
      '''
      Get a point with granularity and then compute the corrsponding thetas and choose that
      '''
      minX, maxX = min(X.T[1]), max(X.T[1])
      self.Granularity = abs(maxX - minX) / self.M
      self.Thetas = np.zeros(shape=(self.M + 1, self.N))

      for i in range(0, self.M + 1):
        W = _WMatrix(self, np.array([1.0, minX + i*self.Granularity]))
        self.Thetas[i] = np.squeeze((_Update(self, W)))
        self.returnValue.append((minX + i*self.Granularity, self.Thetas[i]))
    else:
      self.Thetas = np.zeros(shape=(self.N, 1))
      self.Thetas = _Update(self)
      self.returnValue = self.Thetas

    return self.returnValue
