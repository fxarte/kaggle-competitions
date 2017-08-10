# tweet tag prediction
#preprocess data

import os
import joblib
import numpy as np 
import pandas as pd
import re
import unicodedata
import collections
import pprint
import time

#Sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report

#nltk
import nltk

_RE_links = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')# links url
_RE_tweet_accounts = re.compile('@\\w+')
_RE_tweet_hashtag = re.compile('#\\w+')
_RE_tweet_emojis = re.compile('[\U0001F602-\U0001F64F]')
_RE_city_airport_code = re.compile('[A-Z]{3,}')

import plot_learning_curve as lc

def get_data(file_path, refresh_cached=False, for_training=True):
    '''
        file_path: path to data in csv format
    '''
    df=None
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(file_path))))
    cached_data_filename = DATA_DIR+'/cached_{}_data.pkl.z'.format('training' if for_training else 'testing')
    if not refresh_cached and os.path.exists(cached_data_filename):
        df = joblib.load(cached_data_filename)
    else:
        print("Reading {} data".format('training' if for_training else 'testing'))
        df = pd.read_csv('../input/Tweets.csv', encoding='utf-8')
        joblib.dump(df, cached_data_filename, compress=3)
    return df


def explore_data(data_frame):
    '''
        frequencies, proirs, and such
    '''
    y_tag=data_frame['negativereason']
    y_tag=y_tag.fillna('__NO_DATA__')
    print(y_tag.value_counts()/y_tag.shape[0])
    # print(y_tag.value_counts()/y_tag.shape[0])


def cleanup_data(data_frame):
    '''
        remove data considered not usefull
    '''
    #Removing unlabelled data
    data_frame = data_frame[pd.notnull(data_frame['negativereason'])]
    # print(data_frame['negativereason'].value_counts())
    # data_frame = data_frame.loc[data_frame['negativereason']!='Damaged Luggage']
    # data_frame = data_frame.loc[data_frame['negativereason']!='longlines']
    # print(data_frame['negativereason'].value_counts())
    return data_frame

NE_TYPES=['ORGANIZATION','PERSON','LOCATION','DATE','TIME','MONEY','PERCENT','FACILITY','GPE']
def ner_data(tweet):
    global NE_TYPES
    annotated = nltk.ne_chunk(tweet)
    tweet=[]
    for c in annotated:
        if hasattr(c,'label'):
            tweet.append(c.label())
        else:
            tweet.append(c[0])
    return tweet

def preprocess_data(X, y=None):
    return [preprocess_tweet(x) for x in X], y
    
def preprocess_tweet(tweet):
    # 2 noise tweet artifacts
    # http://stackoverflow.com/a/28333439/1599129
    # http://stackoverflow.com/a/32988358/1599129
    tweet = _RE_tweet_emojis.sub(lambda m: '__'+unicodedata.name(m.group()).upper()+'__', tweet)
    # print(tweet)
    tweet = _RE_links.sub('__LINK__', tweet)
    NER=False
    if NER==True:
        tweet = nltk.word_tokenize(tweet)
        # tweet = tweet.split()
        tweet = nltk.pos_tag(tweet)
        tweet = ner_data(tweet)
        tweet = " ".join(tweet)
    else:
        tweet = _RE_tweet_accounts.sub('__ORGANIZATION__', tweet)
        tweet = _RE_tweet_hashtag.sub('__HASHTAG__', tweet)
        
    return tweet
    
if __name__=="__main__":
    data_file_path = '../input/Tweets.csv'
    # data_file_path = '../../../chatdesk/Airline-Tags.csv'
    df = get_data(data_file_path)
    # explore_data(df)
    
    #remove unwanted data
    df = cleanup_data(df)
    # explore_data(df)
    
    #prerpocess
    X=df['text'].values
    y=df['negativereason'].values
    X, y = preprocess_data(X, y)
    # for a,b in zip(y[:3], X[:3]):
        # print(a, '  ', " ".join(b))
    import sys
    # sys.exit()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    unique, counts = np.unique(y_train, return_counts=True)
    print(np.asarray((unique, counts)).T)
    
    # http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(stop_words='english')),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(class_weight='balanced', loss='log')),
    ])
    
    categories=['alt.atheism','talk.religion.misc',]
    
    parameters = {
        'vect__max_df': ( 0.75, 1.0), #.75, .75. 1.0
        # 'vect__max_features': (None), #None, None, None
        'vect__ngram_range': ((1, 1), (1, 2), (1, 3)),  # bigram, bigram, bigram
        'tfidf__use_idf': (True, False),#True, True,  True
        'tfidf__norm': ('l1', 'l2'),#l2, l2, l2
        'clf__alpha': (0.00001, 0.000001, 0.000001),#0.0001, 0.0001, 0.0001
        'clf__penalty': ('l2', 'elasticnet'),#l2, 'elasticnet', l2
        'clf__n_iter': (5, 10),#5, 10, 5
    }
    
    learn = "oo"

    if learn=="o":
        grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, cv=5)
        print("Performing grid search...")
        print("pipeline:", [name for name, _ in pipeline.steps])
        print("parameters:")
        pprint.pprint(parameters)
        t0 = time.time()
        grid_search.fit(X_train, y_train)
        print("done in %0.3fs" % (time.time() - t0))
        print()

        print("Best score: %0.3f" % grid_search.best_score_)
        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))
    else:
        ''' traing and predict sample with Best parameters set:
        clf__alpha: 1e-05
        clf__penalty: 'elasticnet'
        vect__max_df: 1.0
        vect__ngram_range: (1, 2)
        # all options
            clf__alpha: 0.0001
            clf__n_iter: 5
            clf__penalty: 'l2'
            tfidf__norm: 'l2'
            tfidf__use_idf: True
            vect__max_df: 1.0
            vect__max_features: None
            vect__ngram_range: (1, 2)
        '''

        best_params = {
            'clf__alpha': 0.0000002# reduce and the likelihood of the class increases
            ,'clf__n_iter': 5
            ,'clf__penalty': 'elasticnet'
            ,'tfidf__norm': 'l2'
            ,'tfidf__use_idf': True
            ,'vect__max_df': 1.0
            ,'vect__max_features': None
            ,'vect__ngram_range': (1, 2)
        }
        pipeline.set_params(**best_params)
        
        
        #Learning curve
        plt = lc.plot_learning_curve(pipeline, "Teewt tag Prediction", X, y, ylim=None, cv=None,
            n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5))
        plt.savefig('TeewtTagPrediction.png')
        sys.exit()
        
        pipeline.fit(X_train, y_train)
        
        print("  training done!")
        scores = cross_val_score(pipeline, X_test, y_test)
        score = scores.mean()
        print("Score:{}".format(score)) # Several unbalanced classes, accuracy is not suggested as performance metric
        # using F1
        y_true, y_pred = y_test, pipeline.predict(X_test)
        print(classification_report(y_true, y_pred))
        
        
        sample=['@VirginAmerica everything was fine until you lost my bag']
        sample_x, _ = preprocess_data(sample)

        collections.OrderedDict()
        zpd = zip(pipeline.classes_, pipeline.predict_proba(sample_x)[0])
        print("Tag prediction scores for: '{}'".format(sample))
        for _c , _p in sorted(zpd, key=lambda t: t[1], reverse=True):
            print( _p, _c )
    
    
