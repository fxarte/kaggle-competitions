from __future__ import print_function
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

import numpy as np
from numpy import genfromtxt, savetxt
import time
import pickle


def save_predictor(predictor, params, score=None):
    print(params)
    classifier=type(predictor).__name__
    base_estimator=''
    if hasattr(predictor, 'base_estimator'):
        print("base_estimator FOUND")
        base_estimator="__{}".format(type(predictor.base_estimator).__name__)
    s = []
    _score = ""
    if score:
        _score = "__{0:.3}".format(score)
    if params:
        for k, v in params.iteritems():
            s.append("{}.{}".format(k, v))
    predictor_filename = "{}{}__{}{}".format(classifier,base_estimator,"..".join(s), _score)
    with open(predictor_filename+'.pkl', 'wb') as output:
        pickle.dump(predictor, output, pickle.HIGHEST_PROTOCOL)
    return predictor_filename

def load_predictor(filename):
    pkl_file = open(filename, 'rb')
    return pickle.load(pkl_file)

def do_cv(full_train=False):
    REFRESH=False
    if REFRESH:
        dataset = genfromtxt(open('train.csv','r'), delimiter=',', dtype='f8')[1:]
        target = [x[0] for x in dataset]
        train = [x[1:] for x in dataset]
        np.savez_compressed("training_data_z", train=train, target=target)
    else:
        data=np.load("training_data_z.npz")
        train=data['train']
        target=data['target']
    print("  Train data loaded: {}".format(len(train)))

    X, X_test, y, y_test = train_test_split(train, target, test_size=0.2, random_state=42)
    param_grid = {
        'n_estimators': [50, 100,200]
        ,'learning_rate': [.1, .5, 1, 2]
        ,'base_estimator__n_estimators': [50, 100,250,500]
        ,'base_estimator__max_depth': [1,2,4,None]
        ,'base_estimator__min_samples_split': [2,4,8]
        ,'base_estimator__bootstrap': [False]
        ,'base_estimator__criterion': ['entropy','gini']
    }
    #BEST{'min_samples_split': 4, 'splitter': 'best', 'criterion': 'entropy', 'max_depth': None, 'min_samples_leaf': 1}
    base_estimator= ExtraTreesClassifier(n_jobs=-1)
    estimator = AdaBoostClassifier(base_estimator=base_estimator)
    searcher = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1)
    searcher.fit(X, y)
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in searcher.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
    print()
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, searcher.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
    print("Best parameters set found on development set:")
    print()
    print(searcher.best_params_)
    print()



def main():
    t = time.time()
    #create the training & test sets, skipping the header row with [1:]
    print("Loadind data")
    target = None
    train = None
    REFRESH=True
    RETRAIN_MODEL=True
    if REFRESH:
        dataset = genfromtxt(open('train.csv','r'), delimiter=',', dtype='f8')[1:]
        target = [x[0] for x in dataset]
        train = [x[1:] for x in dataset]
        np.savez_compressed("training_data_z", train=train, target=target)
    else:
        data=np.load("training_data_z.npz")
        train=data['train']
        target=data['target']
    print("  Train data loaded: {}".format(len(train)))

    if REFRESH:
        test = genfromtxt(open('test.csv','r'), delimiter=',', dtype='f8')[1:]
        np.savez_compressed("test_data_z", test=test)
    else:
        data=np.load("test_data_z.npz")
        test=data['test']
    print("  Test data loaded: {}".format(len(test)))

    clf=None
    filename="submission2"
    X, X_test, y, y_test = train_test_split(train, target, test_size=0.2, random_state=42)
    if RETRAIN_MODEL:
        #create and train the random forest
        base_estimator_best_args={'n_jobs':-1, 'verbose':2, 'n_estimators':100}
        base_estimator = RandomForestClassifier(**base_estimator_best_args)
        # base_estimator_best_args={'n_jobs':-1, 'min_samples_split': 4, 'verbose':0}
        # base_estimator= ExtraTreesClassifier(**base_estimator_best_args)
        clf = AdaBoostClassifier(base_estimator=base_estimator, learning_rate =.01, n_estimators=100)
        print("Fitting model")
        # clf.fit(train, target)
        clf.fit(X, y)
        filename = save_predictor(clf,base_estimator_best_args)
    else:
        clf=load_predictor('RandomForestClassifier__n_estimators.100..n_jobs.-1..verbose.2.pkl')

    print("Score: {}".format(clf.score(X_test,y_test)))
    predictions=clf.predict(test)
    submission_data=np.dstack((np.arange(1, predictions.size+1),predictions))[0]
    savetxt(filename+'.csv', submission_data, header='ImageId,Label', delimiter=',', fmt='%d,%d', comments="")
    print("Predictor fitted in {} secs".format(time.time() - t))
if __name__=="__main__":
    # do_cv()
    main()
