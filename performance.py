from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

def plot_print(regr, X, y):
  # The coefficients
  print('Coefficients: \n', regr.coef_)
  # The mean square error
  # print("Residual sum of squares: %.2f"% np.mean((regr.predict(X) - y) ** 2))
  # Explained variance score: 1 is perfect prediction
  # print('Variance score: %.2f' % regr.score(X, y))

  # Plot outputs
  # plt.scatter(X, y,  color='black')
  # plt.plot(X, regr.predict(X), color='blue',linewidth=3)
  # plt.show()

def main(argv):
  X=np.array([[54,66,120],
  [65,175,240],
  [149,330,479],
  [318,640,958]])

  y1=[15,31,138,432]
  y2=[29,92,376,1613]
  
  first_stage=linear_model.LinearRegression()
  first_stage.fit(X, y1)

  regr_total_time = linear_model.LinearRegression()
  regr_total_time.fit(X, y2)
  regr_total_from_firststage_time = linear_model.LinearRegression()
  A=np.array([[15],[31],[138],[432]])
  B=np.array([29,92,376,1613])
  regr_total_from_firststage_time.fit(A, B)


  
  # plot_print(regr_total_time, X, y1)
  all=np.array([[2323,3312,5635]])
  pred_1_stage=first_stage.predict(all)
  pred_total_time=regr_total_time.predict(all)
  pred_regr_total_from_firststage_time=regr_total_from_firststage_time.predict(1)
  print("First Stage: "+str(pred_1_stage))
  print("Full time time: "+str(pred_total_time))
  print("Full time from first Stage: "+str(pred_regr_total_from_firststage_time))

if __name__ == '__main__':
  from sys import argv
  main(argv)