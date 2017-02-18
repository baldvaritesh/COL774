class Reader():
  '''
  Creates a Reader class object based on the question part
  '''

  def __init__(self, part='a', use='train'):
    self.Part = part
    self.Use = use

  def read(self):
    '''
    Gives the data in the suitable format
    '''

    def partA(self):
      f = open('../../../data/q1/' + self.Use + '.txt', 'r')
      s = [x.split() for x in list(filter(None, f.read().split('\n')))]
      f.close()
      return 


