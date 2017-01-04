import os
import sys
from os.path import dirname
from sys import argv, exit, stderr
import ntpath
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import warnings

def read_in_chunks(file_path, chunksize = 25000, is_test=True):
    filename = ntpath.basename(file_path)
    parent_folder = dirname(file_path)
    X=None
    y=None
    ids=None
    if os.path.exists(file_path+'.npz'):
        print("loading processed data")
        data=np.load(file_path+'.npz')
        # X   = pd.read_pickle(parent_folder + '/' + filename+'_X.pkl')
        # y   = pd.read_pickle(parent_folder + '/' + filename+'_y.pkl')
        # ids = pd.read_pickle(parent_folder + '/' + filename+'_ids.pkl')

        X   = data['X']
        if 'y' in data:
            y   = data['y']
        ids = data['ids']
        print("                       ... loaded!!")
    else:
        print("Processing data")
        #chunksize = 25000
        #9223372036854775807
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
            np.savez_compressed(file_path, X=X, ids=ids)
        else:
            mode = "Training"
            for chunk in pd.read_csv(file_path, chunksize=chunksize, dtype='float32'):
                print("Reading next chunk {} of {}".format(c, len(chunk)))
                ids_chunks.append(chunk.ix[:,0])
                y_chunks.append(chunk.ix[:,-1])
                X_chunks.append(chunk.ix[:,1:chunk.shape[1]-1])
            X = pd.concat(X_chunks, axis=0)
            X = X.replace(np.nan,0).values
            y = pd.concat(y_chunks, axis=0).values
            ids = pd.concat(ids_chunks, axis=0).values
            np.savez_compressed(file_path, X=X, y=y, ids=ids)
        c+=1

        print("{} Data loaded: {} items from: '{}'".format(mode, len(ids), file_path))
        # X.to_pickle(parent_folder + '/' + filename+'_X.pkl')
        # y.to_pickle(parent_folder + '/' + filename+'_y.pkl')
        # ids.to_pickle(parent_folder + '/' + filename+'_ids.pkl')
    return ids, X, y

def learn(X, y, predictor_name):
    # print(y.shape)
    # return None
    if predictor_name:
        print("Loading from pkl file")
        predictor = joblib.load(predictor_name)
    else:
        # BE = DecisionTreeClassifier(max_depth=None, random_state=0, class_weight={1:0.00588023520941})
        clf_args = {'class_weight':{1:0.00588023520941}, 'n_jobs':-1, 'n_estimators': 25, 'min_samples_split': 8, 'max_depth': None, 'bootstrap': False}
        BE = ExtraTreesClassifier(**clf_args)
        predictor = AdaBoostClassifier(base_estimator=BE, random_state=0, learning_rate=0.5 )

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        predictor.fit(X_train, y_train)
        predictions = predictor.predict(X_test)

        from sklearn.metrics import matthews_corrcoef
        print("MCC: {}".format(matthews_corrcoef(y_test, predictions)))
        joblib.dump(predictor, type(predictor).__name__+".pkl", compress=1);
    return predictor

def predict(predictor, X):
    predictions = predictor.predict(X)
    return predictions

def writeoutput(outputfilename, predictions, ids):
    with open(outputfilename, 'w') as text_file:
        text_file.write("Id,Response\n")
        for i in range(0,len(predictions)):
            # print(int(ids.loc[i]))
            id = int(ids[i])
            pred = int(predictions[i])
            text_file.write("{},{}\n".format(id, pred))

def best_parameter_search(ids, X, y):
    from sklearn.externals import joblib
    
    warnings.filterwarnings("ignore")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # This workaround addresses some memory consumption issues: See http://stackoverflow.com/a/24411581/1599129
    some_filename = "GridSearchCV_X_train.data"
    joblib.dump(X_train, some_filename)
    X_train = joblib.load(some_filename, mmap_mode='r+')

    tuning_parameters = [
        {'n_estimators': [10,25,50,100,200], 'max_depth':[2,4,None],'min_samples_split':[2,4,8],'bootstrap': [False]}
    ]
    clf = GridSearchCV(ExtraTreesClassifier(n_jobs=2, warm_start=True, class_weight={1:0.00588023520941}), tuning_parameters, cv=3, verbose=2, n_jobs=3)

    clf.fit(X_train, y_train)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
      print("%0.3f (+/-%0.03f) for %r"
            % (mean_score, scores.std() * 2, params))
    print()
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
    '''
    Best for  ExtraTreesClassifier: {'n_estimators': 25, 'min_samples_split': 8, 'max_depth': None, 'bootstrap': False}
    '''


if __name__=="__main__":
    ids, X, y = read_in_chunks(argv[1], is_test=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    Operation = 'explore'
    if Operation == 'explore':
        best_parameter_search(ids, X, y)
    else:
        predictor_name = "DecisionTreeClassifier.pkl"
        predictor_name = "ExtraTreesClassifier.pkl"
        predictor_name = ""
        print("Training")
        predictor = learn(X,y, predictor_name)
        predictor_name = type(predictor).__name__
        predictor_name = "AdaBoostClassifier_ExtraTreesClassifier_"
        print("         ... New predictor trained: {}".format(predictor_name))

        ids, X, _ = read_in_chunks(argv[2])
        print("Prediciting")
        predictions = predict(predictor, X)
        print("            ... done!")
        
        print("Writing predictions!")
        writeoutput(predictor_name+'.txt', predictions, ids)
        print("            ... done!")
