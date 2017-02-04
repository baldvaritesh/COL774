import numpy as np
import collections
from src.models.common import *

class GaussianDiscriminant:
  '''
  Creates a GDA object
  '''

  def __init__(self, sameCov = True):
    '''
    Create the parameters needed for the object
    '''
    self.U = None
    self.Cov = None
    self.Phi = None
    self.SameCov = sameCov
    self.returnValue = None

  def Fit(self, X, y):
    '''
    Here we estimate the ML parameters from the dataset
    '''

    def _phi(self):
      nonlocal y
      self.Mapping = list(set(y.T[0]))
      return (collections.Counter(y.T[0]))[self.Mapping[1]] * 1.0 / self.M

    def _u(self, idx):
      nonlocal X, y
      numR = np.zeros(shape=(self.N))
      for i in range(0, self.M):
        if(y[i] == self.Mapping[idx]):
          numR += X[i]
      denR = (collections.Counter(y.T[0]))[self.Mapping[idx]]
      return numR / denR

    def _Cov(self, idx):
      nonlocal X, y
      cov = np.zeros(shape=(self.N, self.N))
      '''
      in case the covariance matrix is assumed to be true
      '''
      if(self.SameCov == True):
        for i in range(0, self.M):
          uI = self.U[self.Mapping.index(y[i])]
          cov += np.outer((X[i] - uI), (X[i] - uI))
        return cov / self.M
      '''
      When separate covariance matrices exist
      '''
      for i in range(0, self.M):
        if(y[i] == self.Mapping[idx]):
          uI = self.U[self.Mapping.index(y[i])]
          cov += np.outer((X[i] - uI), (X[i] - uI))
      denR = (collections.Counter(y.T[0]))[self.Mapping[idx]]
      return cov / denR

    (X, y) = PreProcessFit(X, y, model='gaussianDiscriminant')
    self.M = len(X)
    self.N = len(X.T)
    '''
    Find all the parameters
    '''
    self.Phi = _phi(self)

    self.U = []
    for i in range(0, len(self.Mapping)):
      self.U.append(_u(self,i))
    self.U = np.array(self.U)

    self.Cov = []
    if(self.SameCov == True):
      self.Cov.append(_Cov(self, 0))
      self.Cov.append(self.Cov[0])
    else:
      self.Cov.append(_Cov(self, 0))
      self.Cov.append(_Cov(self, 1))
    self.Cov = np.array(self.Cov)

    return self.returnValue


  def DecisionBoundary(self):

    def _constantTerm(self):
      a = np.log(self.Phi / (1 - self.Phi))
      b = 0.5 * np.log(np.linalg.det(self.Cov[0]) / np.linalg.det(self.Cov[0]))
      c = 0.5 * ((np.dot(self.U[0].T, np.dot(self.InvCov[0], self.U[0]))) - (np.dot(self.U[1].T, np.dot(self.InvCov[1], self.U[1]))))
      return a + b + c

    def _linearTerm(self):
      return (np.dot(self.U[1].T, self.InvCov[1]) - np.dot(self.U[0].T, self.InvCov[0]))

    def _quadraticTerm(self):
      return 0.5 * (self.InvCov[0] - self.InvCov[1])

    def _linearDecisionBoundary(self):
      return (_constantTerm(self), _linearTerm(self))

    def _quadraticDecisionBoundary(self):
      '''
      Note the difference in the way quadratic term is presented compared to the linear term
      '''
      return (_constantTerm(self), _linearTerm(self), _quadraticTerm(self))

    '''
    Create Inverse Matrices since they are used for decision boundary
    '''
    self.InvCov = []
    for i in self.Cov:
      self.InvCov.append(np.linalg.inv(i))
    self.InvCov = np.array(self.InvCov)

    if(self.SameCov == True):
      return _linearDecisionBoundary(self)
    return _quadraticDecisionBoundary(self)


