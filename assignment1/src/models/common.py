import numpy as np
'''
Utility calls for all the models and analysis
'''

def PreProcessFit(X,y):
	'''
	Shape for X should be: [nsamples nfeatures]
	Shape for y should be: [nsamples nlabels]
	'''
	if(type(X) != 'np.ndarray'):
		X = np.array(X)
	if(type(y) != 'np.ndarray'):
		y = np.array(y)

	if(len(y.shape) == 1 or len(X.shape) == 1):
		print('Shapes for input data not correctly aligned')
		return (None, None)

	X = np.c_[np.ones(shape=(len(y))), X]
	return (X, y)

def PreProcessPredict(X, modelFeatures):
	'''
	Shape for X should be: [nsamples nfeatures]
	modelFeatures: number of parameters for the model
	'''
	y = len(X)
	if(type(X) != 'np.ndarray'):
		X = np.array(X)

	if(len(X.shape) == 1):
		print('Shapes for input data not correctly aligned')
		return None
	X = np.c_[np.ones(shape=(len(y))), X]
	if(modelFeatures != len(X.T)):
		print('Number of features for model and data do not match')
		return None
	return X
