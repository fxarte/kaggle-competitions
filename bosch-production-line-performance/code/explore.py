from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report

import os
import sys
from os.path import dirname
from sys import argv, exit, stderr
import ntpath
import numpy as np
import pandas as pd
import joblib
from time import time
t0 = time()

def save_cached_data(data_set, data_set_name=None):
    if not data_set_name:
        data_set_name="{}".format(type(data_set).__name__)
    joblib.dump(data_set, data_set_name, compress=3);
    
def load_cached_data(data_set_name):
    return joblib.load(data_set_name)

    
def load_data(file_path, chunksize = 25000, is_test=True):
    filename = ntpath.basename(file_path)
    parent_folder = dirname(file_path)
    X=None
    y=None
    ids=None
    cached_data_filename = parent_folder+'/cached_training_data_'+filename+'.data_set.pkl.z'
    if os.path.exists(cached_data_filename):
        print("loading processed data")
        data=load_cached_data(cached_data_filename)
        X   = data['X']
        if 'y' in data:
            y   = data['y']
        ids = data['ids']
        print("                       ... loaded!!")
    else:
        print("Processing data")
        chunks = []
        X_chunks = []
        y_chunks = []
        ids_chunks = []

        c=0
        mode = "Testing"
        if is_test:
            for chunk in pd.read_csv(file_path, chunksize=chunksize, dtype='float32'):
                print("Reading next chunk {} of {}".format(c, len(chunk)))
                ids_chunks.append(chunk.ix[:,0])
                X_chunks.append(chunk.ix[:,1:chunk.shape[1]])
            X = pd.concat(X_chunks, axis=0)
            X = X.replace(np.nan,0).values
            ids = pd.concat(ids_chunks, axis=0).values
            # np.savez_compressed(file_path, X=X, ids=ids)
            data={'X':X, 'y':None, 'ids':ids}
            save_cached_data(data, data_set_name=cached_data_filename)
        else:
            mode = "Training"
            for chunk in pd.read_csv(file_path, chunksize=chunksize, dtype='float32'):
                print("Reading next chunk {}".format(c))
                ids_chunks.append(chunk.ix[:,0])
                y_chunks.append(chunk.ix[:,-1])
                X_chunks.append(chunk.ix[:,1:chunk.shape[1]-1])
                c+=1
            X = pd.concat(X_chunks, axis=0)
            # This workaround addresses some memory consumption issues: See http://stackoverflow.com/a/24411581/1599129
            # some_filename = "some_filename.joblib.load.data"
            # if not os.path.exists(some_filename):
                # joblib.dump(X, some_filename)
            # X = joblib.load(some_filename, mmap_mode='r+')
            X = X.replace(np.nan,0)
            X=X.values
            y = pd.concat(y_chunks, axis=0).values
            ids = pd.concat(ids_chunks, axis=0).values
            # np.savez_compressed(file_path, X=X, y=y, ids=ids)
            data={'X':X, 'y':y, 'ids':ids}
            save_cached_data(data, data_set_name=cached_data_filename)


        print("{} Data loaded: {} items from: '{}'".format(mode, len(ids), file_path))
    return ids, X, y

def learn(X, y):
    # BE = DecisionTreeClassifier(max_depth=None, random_state=0, class_weight={1:0.00588023520941})
    clf_args = {'class_weight':'balanced', 
        'n_jobs':-1, 
        'n_estimators': 25, 
        'min_samples_split': 8, 
        'max_depth': None, 
        'bootstrap': False
    }
    BE = ExtraTreesClassifier(**clf_args)
    predictor = AdaBoostClassifier(base_estimator=BE, random_state=0, learning_rate=0.5 )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    predictor.fit(X_train, y_train)
    predictions = predictor.predict(X_test)

    from sklearn.metrics import matthews_corrcoef
    print("MCC: {}".format(matthews_corrcoef(y_test, predictions)))
    joblib.dump(predictor, type(predictor).__name__+".pkl", compress=1);
    return predictor
if __name__=="__main__":
    ids, X, y = load_data('../input/train_numeric.csv.zip', chunksize = 25000, is_test=False)
    print("Loading data done in %0.3fs" % (time() - t0))
    # learn(X, y)
    print("Training done in %0.3fs" % (time() - t0))