import numpy as np
from src.models.common import *

class LinearRegression():
  '''
  Creates a LinearRegression object
  '''

  def __init__(self, optimized = False, storeCostFunction = False, eta = 0.00001):
    '''
    Declare the empty parameter set on initialisation
    1. optimized: Optimized takes care for the Barzilai-Borwein method of update for learning rate
    2. storeCostFunction: Takes care for plotting the cost function
    3. returnValue: for doing the parts of assignment
    '''
    self.Thetas = []
    self.PreviousThetas = []
    self.Optimized = optimized
    self.StoreCostFunction = storeCostFunction
    self.returnValue = None
    self.Eta = eta

  def Fit(self, X, y):

    def _CostFunction(thetas):
      nonlocal X, y
      matrix = np.dot(X, thetas) - y
      return 0.5 * np.dot(matrix.T, matrix) / self.M

    def _Gradient(thetas):
      nonlocal X, y
      return (np.dot(X.T, np.dot(X, thetas)) - np.dot(X.T, y)) / self.M

    def _SetStepSize(thetas, previousThetas):
      if(len(previousThetas) == 0):
        return 0.00001
      numR = np.dot((thetas - previousThetas).T , (_Gradient(thetas) - _Gradient(previousThetas)))
      denR = (np.linalg.norm(_Gradient(thetas) - _Gradient(previousThetas)))**2
      return numR / denR

    def _Update(self):
      self.PreviousThetas = self.Thetas.copy()
      self.Thetas -= self.Eta * _Gradient(self.Thetas)

    def _Converged(self):
      if (len(self.PreviousThetas) == 0):
        return False
      if (abs(_CostFunction(self.Thetas) - _CostFunction(self.PreviousThetas)) < 1e-12):
        return True
      return False

    (X, y) = PreProcessFit(X, y, model='regression')
    self.M = len(X)
    self.N = len(X.T)
    '''
    Parameters for the model
    '''
    self.Thetas = np.zeros(shape=(self.N, 1))
    if(self.Optimized):
      self.Eta = _SetStepSize(self.Thetas, self.PreviousThetas)

    if(self.StoreCostFunction == True):
      self.returnValue = []

    count = 0
    while (not _Converged(self)):
      count += 1

      if (self.StoreCostFunction == True) and (count % 10000 == 0):
        self.returnValue.append([count, self.Thetas[0], self.Thetas[1], _CostFunction(self.Thetas)])

      _Update(self)

    if(self.StoreCostFunction == True):
      self.returnValue.append([count, self.Thetas[0], self.Thetas[1], _CostFunction(self.Thetas)])

    return self.returnValue

  def Predict(self, X):
    X = PreProcessPredict(X)
    return np.dot(X, self.Thetas)
