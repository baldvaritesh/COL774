import numpy as np
from src.models.common import *

class LogisticRegression:
	'''
	Creates a Logistic Regression Object
	'''

	def __init__(self):
		self.PreviousThetas = []
		self.Thetas = []
		self.returnValue = None

	def __Sigmoid(self, z):
		'''
		Logistic Function is private to the class
		'''
		return 1.0/(1 + np.exp(-z))

	def Fit(self, X, y):

		def _SigmoidMatrix(self, thetas):
			'''
			The Matrix A in the _StepSize comments
			'''
			nonlocal X
			A = np.zeros(shape=(self.M, self.M))
			for i in range(0, self.M):
				valSigmoid = self.__Sigmoid(np.dot(X[i], thetas))
				A[i][i] = valSigmoid * (1 - valSigmoid)
			return A

		def _SigmoidVector(self, thetas):
			'''
			The vector p in the _StepSize comments
			'''
			nonlocal X
			p = np.zeros(shape=(self.M, 1))
			for i in range(0, self.M):
				p[i] = self.__Sigmoid(np.dot(X[i], thetas))
			return p

		def _Hessian(self, thetas):
			'''
			Taking care of the negative sign while updating
			Value is -((X.T)AX)
			'''
			nonlocal X
			return (np.dot(X.T, np.dot(_SigmoidMatrix(self, thetas), X)))

		def _StepSize(self, thetas):
			'''
			The following is the calculated Step size
			((X.T).A.X))^(-1)(X.T)(y - p)
			'''
			nonlocal X, y
			invTerm = np.linalg.inv(_Hessian(self, thetas))
			thirdTerm = y - _SigmoidVector(self, thetas)
			return np.dot(invTerm, np.dot(X.T, thirdTerm))

		def _CostFunction(self, thetas):
			nonlocal X, y
			cost = 0
			for i in range(0, self.M):
				argSigmoid = np.dot(X[i], thetas)
				cost += ( (y[i] - 1) * argSigmoid + np.log(self.__Sigmoid(argSigmoid)) )
			return cost[0]

		def _Update(self):
			'''
			Update follows the Newton Raphson method
			'''
			self.PreviousThetas = self.Thetas.copy()
			self.Thetas += _StepSize(self, self.PreviousThetas)

		def _Converged(self):
			if(len(self.PreviousThetas) == 0):
				return False
			if(abs(_CostFunction(self, self.Thetas) - _CostFunction(self, self.PreviousThetas)) < 1e-12):
				return True
			return False

		(X,y) = PreProcessFit(X,y,model='regression')
		self.M = len(X)
		self.N = len(X.T)
		'''
    Parameters for the model
    '''
		self.Thetas = np.zeros(shape=(self.N, 1))

		count = 0
		while(not _Converged(self)):
			count += 1
			_Update(self)

		return self.returnValue

	def Predict(self, X):

		def _StepFunction(z):
			if(z > 0.5):
				return 1.0
			return 0.0

		X = PrePredict(S, self.N)
		y = np.array([map(self.__Sigmoid, x) for x in np.dot(X, self.Thetas)])
		return np.array([map(_StepFunction, x) for x in y])
