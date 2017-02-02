import numpy as np
from pylab import meshgrid
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

def JTheta(X, y, t0, t1):
  thetas = np.array([[t0],[t1]])
  matrix = np.dot(X, thetas) - y
  return 0.5 * np.dot(matrix.T, matrix) / len(y)

def PlotProcess(X, y):
  X = np.array([np.ones(shape=(len(y))), X])
  X = X.T
  T0 = np.arange(-1.0, 11.0, 0.1)
  T1 = np.arange(-1.0, 11.0, 0.1)
  T0,T1 = np.meshgrid(T0, T1)
  J = np.zeros(shape=T0.shape)
  for i in range(0, len(T0)):
    for j in range(0, len(T1)):
      J[i][j] = JTheta(X, y, T0[i][j], T1[i][j])
  return (T0, T1, J)

def Q1b(X,y,thetas):
  plt.clf()
  plt.scatter(X,y,c=['blue'])
  plt.plot((min(X), max(X)), (thetas[0] + thetas[1]*min(X), thetas[0] + thetas[1]*max(X)), 'red')
  #plt.savefig('../outputs/Q1b.png')
  plt.show()

def Q1c(X, y, answer):
  '''
  Add scatter plots of actual values to show progress

  Snippet below is taken from http://glowingpython.blogspot.in/2012/01/how-to-plot-two-variable-functions-with.html
  '''
  (T0, T1, J) = PlotProcess(X, y)
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  surf = ax.plot_surface(T0, T1, J, rstride=1, cstride=1, cmap=cm.RdBu,linewidth=0, antialiased=False)
  ax.set_xlabel('Theta0 axis')
  ax.set_ylabel('Theta1 axis')
  ax.set_zlabel('J(Theta) axis')
  ax.zaxis.set_major_locator(LinearLocator(10))
  ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
  fig.colorbar(surf, shrink=0.5, aspect=5)
  plt.title('JTheta')
  plt.show()

def Q1d(X, y):
  '''
  Add scatter plots of actual points to show progress
  '''
  (T0, T1, J) = PlotProcess(X, y)
  plt.contour(T0, T1, J, np.arange(0, 50, 3), colors='k', linestyle='solid')
  plt.show()

def Q3b(X, y, thetas):

  def yVal(x):
    '''
    Equation of line: t0 + t1x1 + t2x2 = 0
    '''
    nonlocal thetas
    return (-thetas[0] - thetas[1] * x) / thetas[2]

  XPositive, XNegative = [[],[]], [[],[]]
  for i in range(0, len(y)):
    if(y[i][0] > 0.0):
      XPositive[0].append(X.T[0][i])
      XPositive[1].append(X.T[1][i])
    else:
      XNegative[0].append(X.T[0][i])
      XNegative[1].append(X.T[1][i])
  plt.scatter(XPositive[0], XPositive[1], c=['blue'])
  plt.scatter(XNegative[0], XNegative[1], c=['red'])
  plt.plot((min(X.T[0]), max(X.T[0])), (yVal(min(X.T[0])), yVal(max(X.T[0]))), 'green')
  plt.show()

def Q4(X, y, part, thetas = []):

  def yLinear(x):
    nonlocal thetas
    return (-thetas[0] - thetas[1][0] * x) / thetas[1][1]

  def partC():
    nonlocal X
    plt.plot((min(X.T[0]), max(X.T[0])), (yLinear(min(X.T[0])), yLinear(max(X.T[0]))), 'green')

  def partE():
    print('Hello')

  XA, XC = [[],[]], [[],[]]
  for i in range(0, len(y)):
    if(y[i][0] == 'Alaska'):
      XA[0].append(X.T[0][i])
      XA[1].append(X.T[1][i])
    else:
      XC[0].append(X.T[0][i])
      XC[1].append(X.T[1][i])
  plt.scatter(XA[0], XA[1], c=['blue'])
  plt.scatter(XC[0], XC[1], c=['red'])

  if(part == 'c'):
    partC()

  if(part == 'e'):
    partC()
    partE()

  plt.show()
