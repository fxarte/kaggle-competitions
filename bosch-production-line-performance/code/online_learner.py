from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier

import os
import sys
from os.path import dirname
from sys import argv, exit, stderr
import ntpath
import numpy as np
import pandas as pd
import joblib
import time
from sklearn.metrics import matthews_corrcoef


def save_model(model):
    model_name="{}.model.pkl.z".format(model.__class__.__name__)
    joblib.dump(model, model_name, compress=3);
    
def load_model(model_name):
    return joblib.load(model_name)


def save_cached_data(data_set, data_set_name=None):
    if not data_set_name:
        data_set_name="{}".format(type(data_set).__name__)
    joblib.dump(data_set, data_set_name, compress=3);
    
def load_cached_data(data_set_name):
    return joblib.load(data_set_name)

def progress(cls_name, stats):
    """Report progress information, return a string."""
    duration = time.time() - stats['t0']
    s = "%10s classifier : \t" % cls_name
    s += "%(n_train)6d train docs (%(n_train_pos)6d positive) " % stats
    # s += "%(n_test)6d test docs (%(n_test_pos)6d positive) " % test_stats
    s += "accuracy: %(accuracy).3f " % stats
    s += "matthews_corrcoef: %(matthews_corrcoef).3f " % stats
    s += "in %.2fs (%5d docs/s)" % (duration, stats['n_train'] / duration)
    return s


    
def load_test_data():
    X=None
    y=None
    ids=None
    file_path = '../input/test_numeric.csv.zip'
    X_chunks = []
    ids_chunks = []
    chunksize=25000
    for chunk in pd.read_csv(file_path, chunksize=chunksize, dtype='float32'):
        print("Reading next {}".format(len(chunk)))
        ids_chunks.append(chunk.ix[:,0])
        X_chunks.append(chunk.ix[:,1:chunk.shape[1]])
    X = pd.concat(X_chunks, axis=0)
    X = X.replace(np.nan,0).values
    ids = pd.concat(ids_chunks, axis=0).values
    return ids, X, y
    
    
def get_matthews_corrcoef(y_test, predictions):
    return matthews_corrcoef(y_test, predictions)

def oo_learn():
    file_path = '../input/train_numeric.csv.zip'
    filename = ntpath.basename(file_path)
    parent_folder = dirname(file_path)
    X=None
    y=None
    ids=None
    print("Online Learning from '{}'".format(file_path))
    chunks = []
    X_chunks = []
    y_chunks = []
    ids_chunks = []
    c=0
    cls = SGDClassifier(n_jobs=7, penalty='l1', random_state=42, learning_rate ='optimal' )
    cls_stats = {}
    
    cls_name='SGDClassifier'
    stats = {'n_train': 0, 'n_train_pos': 0,
             'accuracy': 0.0, 'matthews_corrcoef': 0.0, 'accuracy_history': [(0, 0)], 't0': time.time(),
             'runtime_history': [(0, 0)], 'total_fit_time': 0.0}
    cls_stats[cls_name] = stats
    chunksize=25000
    all_classes = np.array([0, 1])
    tick = time.time()
    total_vect_time = 0.0
    for chunk in pd.read_csv(file_path, chunksize=chunksize, dtype='float32'):
        print("Chunk {}".format(c))
        ids_chunks.append(chunk.ix[:,0])
        y = chunk.ix[:,-1]
        X = chunk.ix[:,1:chunk.shape[1]-1]
        X = X.replace(np.nan,0)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        cls.partial_fit(X_train, y_train, classes=all_classes)
        c+=1
        # accumulate test accuracy stats
        cls_stats[cls_name]['total_fit_time'] += time.time() - tick
        cls_stats[cls_name]['n_train'] += X_train.shape[0]
        cls_stats[cls_name]['n_train_pos'] += sum(y_train)
        tick = time.time()
        cls_stats[cls_name]['accuracy'] = cls.score(X_test, y_test)
        predictions = cls.predict(X_test)
        cls_stats[cls_name]['matthews_corrcoef'] = get_matthews_corrcoef(y_test, predictions)
        cls_stats[cls_name]['prediction_time'] = time.time() - tick
        acc_history = (cls_stats[cls_name]['accuracy'], cls_stats[cls_name]['n_train'])
        cls_stats[cls_name]['accuracy_history'].append(acc_history)
        run_history = (cls_stats[cls_name]['accuracy'], total_vect_time + cls_stats[cls_name]['total_fit_time'])
        cls_stats[cls_name]['runtime_history'].append(run_history)

        print(progress(cls_name, cls_stats[cls_name]))
    save_model(cls)
    return cls
def writeoutput(outputfilename, predictions, ids):
    with open(outputfilename, 'w') as text_file:
        text_file.write("Id,Response\n")
        for i in range(0,len(predictions)):
            # print(int(ids.loc[i]))
            id = int(ids[i])
            pred = int(predictions[i])
            text_file.write("{},{}\n".format(id, pred))

if __name__=="__main__":
    # estimator = oo_learn()
    estimator = load_model('SGDClassifier.model.pkl.z')
    ids, X, _ = load_test_data()
    predictions = estimator.predict(X)
    writeoutput(estimator.__class__.__name__+'_predictions.txt', predictions, ids)
