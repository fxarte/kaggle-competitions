from __future__ import print_function

import os
import sys
import re
import train as my_train
import cv2
import numpy as np
import pickle

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

data_dir='G:/data/ultrasound-nerve-segmentation'

def load_data(files_folder='train', refresh=False):
  # print("processing: {}".format(files_folder))
  my_train.load_csv_file()
  features=[]
  targets=[]
  if refresh:
    for subject_id in range(1, 48):
      for file in [f for f in os.listdir(data_dir+"/"+files_folder) if re.match(r'{}_[0-9]+\.tif'.format(subject_id), f)]:
        filename, file_extension = os.path.splitext(file)
        x1,y1,w,h = my_train.get_mask_box(subject_id,filename)
        pos_img_path=data_dir+"/"+files_folder+"/"+file
        img=cv2.imread(pos_img_path,0)
        features.append(img.ravel())
        # print(pos_img_path)
        # print(img.shape)
        # break
        # sys.stdout.write(".")
        if x1:
          targets.append(1)
        else:
          targets.append(0)
    features = np.array(features)
    targets = np.array(targets)
    np.savez_compressed("training_data_z", features=features, targets=targets)
  else:
    data=np.load("training_data_z.npz")
    print(data.keys())
    features=data['features']
    targets=data['targets']
  return features,targets

def get_predictor_filename(classifier, params, score=None):
  # print(params)
  s = []
  _score = ""
  if score:
    _score = "__{0:.3}".format(score)
  if params:
    for k, v in params.iteritems():
      s.append("{}.{}".format(k, v))
  return "{}__{}{}".format(classifier,"..".join(s), _score)

def learn(images_folder="train"):
  '''
    Generates a predictor object of ExtraTreesClassifier type, using the training data provided in `images_folder`.
    Assumes the algorithm's optimal paramters are known at this point
  '''
  features, targets = load_data(refresh=False)
  # print(features.shape)
  # print(targets.shape)
  X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
  
  # '''
  #Best parameters set found on development set:
  #{'min_samples_split': 4, 'n_estimators': 10, 'max_depth': 4,max_features=0.8 +> 0.69}
  # best_params={'n_jobs':-1, 'verbose':1, 'min_samples_split':4, 'n_estimators':100}
  # predictor = ExtraTreesClassifier(**best_params)
  #
  # predictor = GaussianNB()
  
  # best_params={}
  # predictor = SVC()
  
  best_params={}
  predictor = DecisionTreeClassifier()
  print("Building model with params: {}".format(predictor))
  predictor.fit(X_train, y_train)
  score=predictor.score(X_test,y_test)
  print("Score: {}".format(score))
  
  predictor_filename = get_predictor_filename(type(predictor).__name__, best_params, score)
  with open(predictor_filename+'.pkl', 'wb') as output:
    pickle.dump(predictor, output, pickle.HIGHEST_PROTOCOL)
  
  
  '''
  #CrossValidation
  X=features
  y=targets
  print("Started Cross Validation")
  tuned_parameters = [
    {'criterion ':['gini','entropy' ],'max_depth':[1,2,4,8,None],'min_samples_split':[2,4,8,16], 'max_features':['auto','sqrt','log2',None],'random_state':[42]}
  ]
  clf = GridSearchCV(DecisionTreeClassifier(), param_grid=tuned_parameters, cv=3, n_jobs=4)
  clf.fit(X, y)
  print("Best parameters set found on development set:")
  print(clf.best_params_)
  
  '''
  
  
  
if __name__ == '__main__':
  # estimate_best_params(argv[1])
  predictor = learn()
  # predictions = predict(argv[2], predictor)
  # writeoutput(argv[3], predictions)