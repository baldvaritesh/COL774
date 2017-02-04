import numpy as np
from pylab import meshgrid
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from src.models.common import *
import matplotlib.pyplot as plt

def JTheta(X, y, t0, t1):
  thetas = np.array([[t0],[t1]])
  matrix = np.dot(X, thetas) - y
  return 0.5 * np.dot(matrix.T, matrix) / len(y)

def PlotProcess(X, y):
  X, y = PreProcessFit(X, y, 'regression')
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
  plt.xlabel('Area (Normalised)')
  plt.ylabel('House Prices')
  blue_patch = mpatches.Patch(color='blue', label='Data')
  c,= plt.plot((min(X), max(X)), (thetas[0] + thetas[1]*min(X), thetas[0] + thetas[1]*max(X)), 'red', label='Hypothesis')
  plt.legend(handles=[blue_patch, c])
  plt.savefig('../outputs/Q1/partb.png')

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
  plt.savefig('../outputs/Q1/partc.png')

def Q1cScatter(X, y, answer):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(answer.T[1], answer.T[2], answer.T[3], c=['red'])
  ax.set_xlabel('Theta0')
  ax.set_ylabel('Theta1')
  ax.set_zlabel('J(Theta)')
  red_patch = mpatches.Patch(color='red', label='Error Function')
  plt.legend(handles=[red_patch])
  plt.savefig('../outputs/Q1/partc2.png')

def Q1d(X, y, answer, part):
  '''
  Add scatter plots of actual points to show progress
  '''
  (T0, T1, J) = PlotProcess(X, y)
  plt.contour(T0, T1, J, np.arange(0, 50, 3), colors='k', linestyle='solid')
  plt.scatter(answer.T[1], answer.T[2], c=['red'])
  e = mlines.Line2D([], [], color='black', label='Contours')
  red_patch = mpatches.Patch(color='red', label='Error Function')
  plt.legend(handles=[e, red_patch])
  plt.savefig('../outputs/Q1/part'+part+'.png')

def Q2(X, y, part, thetas, intervals = []):
  plt.clf()
  plt.scatter(X, y, c=['blue'])
  plt.xlabel('X')
  plt.ylabel('Y')
  blue_patch = mpatches.Patch(color='blue', label='Data')
  legendHandles = [blue_patch]

  if(part == 'a'):
    c,= plt.plot((min(X), max(X)), (thetas[0] + thetas[1]*min(X), thetas[0] + thetas[1]*max(X)), 'red', label='Unweighted')
    legendHandles.append(c)

  else:
    plt.scatter(X, y, c=['blue'])
    i = 0
    for interval in intervals:
      c,= plt.plot((interval[0][0], interval[0][1]), (thetas[i][0] + thetas[i][1]*interval[0][0], thetas[i][0] + thetas[i][1]*interval[0][1]), 'red', label='Weighted')
      if(i == 0):
        legendHandles.append(c)
      i += 1
  plt.legend(handles=legendHandles)
  plt.savefig('../outputs/Q2/part' + part + '.png')

def Q3(X, y, part, thetas = []):

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
  plt.xlabel('Feature 0')
  plt.ylabel('Feature 1')
  plt.scatter(XPositive[0], XPositive[1], c=['blue'])
  plt.scatter(XNegative[0], XNegative[1], c=['red'])
  blue_patch = mpatches.Patch(color='blue', label='Positive')
  red_patch = mpatches.Patch(color='red', label='Negative')
  legendHandles = [red_patch, blue_patch]
  if(part == 'b'):
    c, = plt.plot((min(X.T[0]), max(X.T[0])), (yVal(min(X.T[0])), yVal(max(X.T[0]))), 'green', label='Linear')
    legendHandles.append(c)
  plt.legend(handles=legendHandles)
  plt.savefig('../outputs/Q3/part' + part + '.png')

def Q4(X, y, part = None, thetas = []):

  def yLinear(x):
    nonlocal thetas
    '''
    Form of thetas: Constant, 1D array
    '''
    return (-thetas[0] - thetas[1][0] * x) / thetas[1][1]

  def yQuadratic(x, y):
    '''
    Form of thetas: Constant, 1D array, 2D array
    Just solving a quadratic equation
    '''
    nonlocal thetas
    alpha = thetas[2][1][1]
    beta = 2 * thetas[2][0][1] * x + thetas[1][1]
    gamma = thetas[2][0][0] * (x ** 2) + thetas[1][0] * x + thetas[0]
    return alpha * (y ** 2) + beta * (y) + gamma


  def partC():
    nonlocal X
    global legendHandles
    c, = plt.plot((min(X.T[0]), max(X.T[0])), (yLinear(min(X.T[0])), yLinear(max(X.T[0]))), 'green', label='Linear')
    return c

  def partE():
    x = np.linspace(-3, 3, 300)
    y = np.linspace(-4, 4, 400)
    x, y = np.meshgrid(x, y)
    plt.contour(x, y, yQuadratic(x, y), [0], colors='k')
    e = mlines.Line2D([], [], color='black', label='Quadratic')
    return e

  XA, XC = [[],[]], [[],[]]
  for i in range(0, len(y)):
    if(y[i][0] == 'Alaska'):
      XA[0].append(X.T[0][i])
      XA[1].append(X.T[1][i])
    else:
      XC[0].append(X.T[0][i])
      XC[1].append(X.T[1][i])
  blue_patch = mpatches.Patch(color='blue', label='Alaska')
  red_patch = mpatches.Patch(color='red', label='Canada')
  legendHandles = [red_patch, blue_patch]
  plt.xlabel('Feature 0')
  plt.ylabel('Feature 1')
  plt.scatter(XA[0], XA[1], c=['blue'])
  plt.scatter(XC[0], XC[1], c=['red'])

  if(part == 'c'):
    legendHandles.append(partC())

  if(part == 'e'):
    legendHandles.append(partC())
    legendHandles.append(partE())

  plt.legend(handles=legendHandles)
  plt.savefig('../outputs/Q4/part' + part + '.png')
