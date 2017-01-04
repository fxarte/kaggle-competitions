from sys import argv
import numpy as np
import pandas as pd
import load_data as my
import time
import matplotlib.pyplot as plt
import joblib
import os
#Classifiers
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB

from sklearn.cross_validation import train_test_split
from sklearn.metrics import matthews_corrcoef

def progress(cls_name, stats, test_stats):
    """Report progress information, return a string."""
    duration = time.time() - stats['t0']
    s = "%20s classifier : \t" % cls_name
    s += "%(n_train)6d train docs (%(n_train_pos)6d positive) " % stats
    s += "%(n_test)6d test docs (%(n_test_pos)6d positive) " % test_stats
    s += "accuracy: %(accuracy).3f " % stats
    s += "in %.2fs (%5d docs/s)" % (duration, stats['n_train'] / duration)
    return s


def plot_accuracy(x, y, x_legend):
    """Plot accuracy as a function of x."""
    x = np.array(x)
    y = np.array(y)
    plt.title('Classification accuracy as a function of %s' % x_legend)
    plt.xlabel('%s' % x_legend)
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.plot(x, y)

def main(train_file_path, test_file_path=None):
    one_class_weight = 0.00581120796927046
    partial_fit_classifiers = {
        'SGD': SGDClassifier(n_jobs=-1, class_weight={1:one_class_weight}),
        'Perceptron': Perceptron(class_weight={1:one_class_weight}),
        'Gaussian NB': GaussianNB(),
        'Passive-Aggressive': PassiveAggressiveClassifier(class_weight={1:one_class_weight}),
    }

    chunksize = 25000
    rng = np.random.RandomState(42)
    all_classes = np.array([0, 1])
    cls_stats = {}
    total_vect_time = 0.0
    for cls_name in partial_fit_classifiers:
        stats = {'n_train': 0, 'n_train_pos': 0,'prior_p':0, 'size': 0,
                 'accuracy': 0.0, 'accuracy_history': [(0, 0)], 't0': time.time(),
                 'runtime_history': [(0, 0)], 'total_fit_time': 0.0}
        cls_stats[cls_name] = stats

    # test data statistics
    test_stats = {'n_test': 0, 'n_test_pos': 0}
    model_name = "classifiers.pkl"
    cls_stats_name = "cls_stats.pkl"
    display_freq = (len(partial_fit_classifiers)-1) if len(partial_fit_classifiers)>1 else 1
    if not os.path.exists(model_name):
        i=0
        for df in pd.read_csv(train_file_path, chunksize=chunksize, dtype='float32', na_values=""):
            chunk = df.replace(np.nan,0)
            tick = time.time()
            total_vect_time += time.time() - tick
            for cls_name, cls in partial_fit_classifiers.items():
                # do some dimensionality reduction...
                # print("REading next chunk of {}".format(len(chunk)))
                # print(chunk.shape[1])
                y=chunk.ix[:,-1]

                # cls_stats[cls_name]['prior_p'] += y.sum()
                # cls_stats[cls_name]['size'] += y.count()
                X=chunk.ix[:,1:chunk.shape[1]-1]

                # print(X.iloc[0])
                # return
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rng)
                
                test_stats['n_test'] += len(y_test)
                test_stats['n_test_pos'] += sum(y_test)

                cls.partial_fit(X_train, y_train, classes=all_classes)

                # accumulate test accuracy stats
                cls_stats[cls_name]['total_fit_time'] += time.time() - tick
                cls_stats[cls_name]['n_train'] += X_train.shape[0]
                cls_stats[cls_name]['n_train_pos'] += sum(y_train)
                tick = time.time()
                
                y_pred = cls.predict(X_test)
                # cls_stats[cls_name]['accuracy'] = cls.score(X_test, y_test)
                cls_stats[cls_name]['accuracy'] = matthews_corrcoef(y_test, y_pred)

                cls_stats[cls_name]['prediction_time'] = time.time() - tick
                acc_history = (cls_stats[cls_name]['accuracy'], cls_stats[cls_name]['n_train'])

                cls_stats[cls_name]['accuracy_history'].append(acc_history)
                run_history = (cls_stats[cls_name]['accuracy'], total_vect_time + cls_stats[cls_name]['total_fit_time'])
                cls_stats[cls_name]['runtime_history'].append(run_history)

                if i % display_freq == 0:
                    print(progress(cls_name, cls_stats[cls_name], test_stats))
                i+=1
            if i % display_freq == 0:
                print('\n')


        joblib.dump(cls_stats, cls_stats_name, compress=1);
        joblib.dump(partial_fit_classifiers, model_name, compress=1);
    else:
        partial_fit_classifiers = joblib.load(model_name)
        cls_stats = joblib.load(cls_stats_name)

    cls_names = list(sorted(cls_stats.keys()))
    # Plot accuracy evolution
    plt.figure()
    for _class, stats in sorted(cls_stats.items()):
        # Plot accuracy evolution with #examples
        # print("{} mean: {}".format(_class, stats['prior_p']/stats['size']))

        accuracy, n_examples = zip(*stats['accuracy_history'])
        plot_accuracy(n_examples, accuracy, "training examples (#)")
        ax = plt.gca()
        ax.set_ylim((0.8, 1))
    plt.legend(cls_names, loc='best')
    plt.show()

    #Predict
    quiz_data = pd.read_csv(test_file_path, dtype='float32', na_values="")
    ids = quiz_data.ix[:,0]
    quiz_data = quiz_data.ix[:,1:chunk.shape[1]]
    quiz_data = quiz_data.replace(np.nan,0)

    print("Test Data loaded")
    for cls_name, cls in partial_fit_classifiers.items():
        predictions = cls.predict(quiz_data)
        with open(cls_name+'_predictions.txt', 'w') as text_file:
            text_file.write("Id,Response\n")
            for i in range(0,len(predictions)):
                text_file.write("{},{}\n".format(int(ids[i]),predictions[i]))
    print("Predictions complete")


if __name__=="__main__":
    test_file=None
    if argv[2]:
        test_file=argv[2]

    main(argv[1], test_file)
