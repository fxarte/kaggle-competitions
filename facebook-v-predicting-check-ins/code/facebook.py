# -*- coding: utf-8 -*-
import zipfile
import pandas as pd
import matplotlib.pyplot
import pylab

def load_data(type='train'):
  data=None
  if not type or type=='train':
    with zipfile.ZipFile('data/train.csv.zip') as z:
      data = pd.read_csv(z.open('train.csv'), header='infer')
  x = data['x']
  y = data['y']
  matplotlib.pyplot.scatter(x,y)
  matplotlib.pyplot.show()

REFRESH=True
if __name__ == '__main__':
    from sys import argv
    # main(argv)
    load_data()
