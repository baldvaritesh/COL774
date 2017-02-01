'''
If you run any module from the interpreter be sure to add the following
two lines to allow relative imports
'''
import sys
if("/home/ritesh/Desktop/IPython/ml/assignment1" not in sys.path):
  sys.path.insert(0, "/home/ritesh/Desktop/IPython/ml/assignment1")

import numpy as np

def SingleFeatureReader(file, functionType):
  f = open(file, 'r')
  s = list(filter(None, f.read().split('\n')))
  f.close()
  s = [np.array([functionType(x)]) for x in s]
  return np.array(s)

def MultipleFeatureReader(file, functionType):
  f = open(file, 'r')
  s = [x.split() for x in list(filter(None, f.read().split('\n')))]
  f.close()
  s = np.array([ np.array(list(map(functionType, x))) for x in s])
  return s

FileNames = {
  'q1' : ["../../data/q1x.dat", "../../data/q1y.dat"], 
  'q2' : ["../../data/q3x.dat", "../../data/q3y.dat"],
  'q3' : ["../../data/q2x.dat", "../../data/q2y.dat"],
  'q4' : ["../../data/q4x.dat", "../../data/q4y.dat"],
}

def Reader(qName, Ftypes, ReaderType):

  def Normalize(X):
    restoreData = []
    for row in X.T:
      meanX = row.mean()
      stdX = row.std()
      row -= meanX
      row /= stdX
      restoreData.append({'mean': meanX, 'std': stdX})
    return (X, restoreData)

  Data = []
  for i in range(0, len(FileNames[qName])):
    if(ReaderType[i] == 0):
      Data.append(SingleFeatureReader(FileNames[qName][i], Ftypes[i]))
    else:
      Data.append(MultipleFeatureReader(FileNames[qName][i], Ftypes[i]))
  Data[0] = Normalize(Data[0])
  return Data