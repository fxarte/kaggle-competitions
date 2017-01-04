# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 16:23:57 2016

@author: felix
"""

import sys
from time import time
import numpy as np
import pylab as pl
import pickle
import matplotlib.pyplot as plt
from sklearn import cross_validation

sys.path.append("../../ud120-projects/tools/")
from feature_format import featureFormat, targetFeatureSplit



import nltk
import sklearn

print('The nltk version is {}.'.format(nltk.__version__))
print('The scikit-learn version is {}.'.format(sklearn.__version__))
#sys.exit()



def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
  """Helper function to plot a gallery of images"""
  pl.figure(figsize=(1.8 * n_col, 2.4 * n_row))
  pl.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
  for i in range(0,np.shape(images)[0]):
      pl.subplot(n_row, n_col, i + 1)
      pl.imshow(images[i].reshape((h, w)), cmap=pl.cm.gray)
      pl.title(titles[i], size=12)
      pl.xticks(())
      pl.yticks(())

data=np.loadtxt(open("G:/cloud/Dropbox/Felix/projects/Kaggle/train.csv","rb"),delimiter=",",skiprows=1)
confusing=[]
for row in data:
  if row[0] == 9:
    row[0] = 1
    confusing.append(row)
  if row[0] == 4:
    row[0] = 1
  else:
    row[0] = 0

train_labels, train_features = targetFeatureSplit(data)
confusing_train_labels, confusing_train_features = targetFeatureSplit(confusing)
confusing_X_train, confusing_X_test, confusing_y_train, confusing_y_test = cross_validation.train_test_split(confusing_train_features, confusing_train_labels, test_size=0.4, random_state=0)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_features, train_labels, test_size=0.4, random_state=0)
#print np.shape(X_train),np.shape(X_test),np.shape(y_train),np.shape(y_test)
#sys.exit()
#print train_labels
#print train_features[0]
#print np.shape(train_features), len(train_labels)

samples=confusing_train_features[0:9]
titles=confusing_train_labels[0:9]
#print np.shape(samples)[0],np.shape(titles)
#sys.exit()
plot_gallery(samples, titles, 28, 28)


#print your_list[0][0]
#from sklearn.neural_network import MLPClassifier
from sklearn import svm
#clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
k='poly'
clf = svm.SVC(kernel=k)
t0 = time()
clf.fit(X_train, y_train)
print "training time  %0.3fs" % (time() - t0)
#s = pickle.dumps(clf)
print "Zeros digits: kernel ",k, " score: ", clf.score(X_test,y_test)
print "confusing, similar to 4s, kernel ",k, " score: ", clf.score(confusing_X_test,confusing_y_test)

#All digits score kernel=linear: 0.905952380952 training time  81.661s
#All digits score kernel=poly: 0.969583333333 training time  55.751s
#All digits score kernel=rbf: 0.1125 training time  7815.903s
#All digits score kernel=sigmoid: 0.1125 training time  749.934s


