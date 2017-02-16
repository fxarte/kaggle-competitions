# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


#reference: http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html
import sys
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.
from time import time
import re, string
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier

from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer, TweetTokenizer

import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import svm

class DenseTransformer(TransformerMixin):
    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return dict()

class TextCleaner(BaseEstimator, TransformerMixin):
    _numeric = re.compile('(\$)?\d+([\.,]\d+)*')# with decimals and so on figures
    _links = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')# links url
    _trans_table = str.maketrans({key: None for key in string.punctuation})

    def clean_text(self, tokens):
        for i, w in enumerate(tokens):
            if TextCleaner._numeric.match(w):
                tokens[i] = w.translate(TextCleaner._trans_table)
            elif TextCleaner._links.match(w):
                tokens[i] = '_LINK_'
            else:
                continue
        return tokens
            
    def transform(self, X):
        for i, item in enumerate(X):
            tokenized = item.split()
            X[i] = " ".join(self.clean_text(tokenized))

        return X
        
    def fit_transform(self, X, y=None):
        return self.transform(X)
    
    def fit(self, X, y=None):
        self.transform(X)
        return self

def print_significant_features(pipeline=None, n=40):
    feature_names = pipeline.get_params()['vect'].get_feature_names()
    coefs=[]
    if hasattr(pipeline.get_params()['clf'], 'coef_'):
        coefs = pipeline.get_params()['clf'].coef_
    elif hasattr(pipeline.get_params()['clf'], 'feature_importances_'):
        coefs.append(pipeline.get_params()['clf'].feature_importances_)
    else:
        print("Warning: plot_significant_features: Unable to process pipeline at the moment")
        return

    print("Total features: {}".format(len(coefs[0])))
    
    coefs_with_fns = sorted(zip(coefs[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])

    for (coef_2, fn_2) in top:
        print(coef_2, '----', fn_2)
        # print("%.4f: %-16s" % (coef_2, str(fn_2)))

def plot_significant_features(pipeline=None, n=20):
    feature_names = pipeline.get_params()['vect'].get_feature_names()
    coefs=[]
    if hasattr(pipeline.get_params()['clf'], 'coef_'):
        coefs = pipeline.get_params()['clf'].coef_
    elif hasattr(pipeline.get_params()['clf'], 'feature_importances_'):
        coefs.append(pipeline.get_params()['clf'].feature_importances_)
    else:
        print("Warning: plot_significant_features: Unable to process pipeline at the moment")
        return
    
    
    print("Total features: {}".format(len(coefs[0])))
    coefs_with_fns = sorted(zip(coefs[0], feature_names))
    top = coefs_with_fns[:-(n + 1):-1]
    
    y,X = zip(*top)

    plt.figure()
    plt.title("Top 20 most important features")
    plt.gcf().subplots_adjust(bottom=0.25)
    ax = plt.subplot(111)
    
    ax.bar(range(len(X)), y, color="r", align="center")
    ax.set_xticks(range(len(X)))
    ax.set_xlim(-1, len(X))
    ax.set_xticklabels(X,rotation='vertical')
    plt.savefig('sentiment_feature_importance.png')
    plt.close()


def build_sentiment_pipe():
    pipe_params = {
        'tfidf__norm': 'l2'
        ,'tfidf__use_idf': True
        ,'vect__max_df': .7
        ,'vect__max_features': 10000
        ,'vect__ngram_range': (1,2)
        ,'vect__strip_accents': 'unicode'
        ,'vect__stop_words': stops
    }

    pipeline = Pipeline([
        ('cleantxt', TextCleaner()),
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        # ('clf', OneVsRestClassifier(LinearSVC(class_weight='balanced', random_state=42))),
        ('clf', OneVsRestClassifier(ExtraTreesClassifier(n_jobs=-1, n_estimators=500, class_weight='balanced', random_state=42))), #0.7636625287058755
        # ('clf', DecisionTreeClassifier()),#Score:0.6279615220025925
        # ('clf', ExtraTreesClassifier(n_jobs=-1, n_estimators=500, class_weight='balanced', random_state=42)),#Score:0.7634365520206998
        #('dense', DenseTransformer()),
        #('clf', GaussianNB())#Score:0.6259115628872863
    ])

    pipeline.set_params(**pipe_params)
    return pipeline

def build_tagger_pipe(override_estimator=None):
    pipe_params = {
        'tfidf__norm': 'l2'
        ,'tfidf__use_idf': True
        ,'vect__max_df': .7
        ,'vect__max_features': 10000
        ,'vect__ngram_range': (1,2)
        ,'vect__strip_accents': 'unicode'
        ,'vect__stop_words': stops
    }

    DT = DecisionTreeClassifier(class_weight='balanced')
    Adab_DT = AdaBoostClassifier(base_estimator=DT)
    ET = ExtraTreesClassifier(n_jobs=-1, n_estimators=500, class_weight='balanced', random_state=42)
    OneVsAll_ET = OneVsRestClassifier(ET)
    OneVsAll_DT = OneVsRestClassifier(DT)
    DT_educated = DecisionTreeClassifier(
                class_weight='balanced', 
                max_depth= 8,
                min_samples_split= 2,
                criterion='entropy',
                splitter='random'
    )
    OneVsAll_DT_educated = OneVsRestClassifier(DT_educated)
    Adab_ET = AdaBoostClassifier(base_estimator=ET)
    Adab_OvA_DT = AdaBoostClassifier(base_estimator=OneVsAll_DT)
    SVC = svm.SVC()
    LSVC = svm.LinearSVC()
    SVC_ovo = svm.SVC(decision_function_shape='ovo')
    SVC_ovr = svm.SVC(decision_function_shape='ovr')
    OneVsAll_SVC = OneVsRestClassifier(SVC)
    
    estimator = LSVC if override_estimator is None else override_estimator
    
    params = [
        ('cleantxt', TextCleaner()),
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        # ('clf', OneVsRestClassifier(LinearSVC(class_weight='balanced', random_state=42))),
        ('clf', estimator),
        # ('clf', OneVsRestClassifier(ExtraTreesClassifier(n_jobs=-1, n_estimators=500, class_weight='balanced', random_state=42))), #0.7636625287058755
        # ('clf', DecisionTreeClassifier()),#Score:0.6279615220025925
        # ('clf', ExtraTreesClassifier(n_jobs=-1, n_estimators=500, class_weight='balanced', random_state=42)),#Score:0.7634365520206998
        # ('dense', DenseTransformer()),('clf', GaussianNB())#Score:0.6259115628872863
        # ('clf', Adab_DT),
        # ('clf', DT),
        # ('clf', Adab_ET),
    ]
    pipeline = Pipeline(params)

    pipeline.set_params(**pipe_params)
    return pipeline

def do_learn():
    df = pd.read_csv('../input/Tweets.csv')
    global stops
    df_tag = df[pd.notnull(df['negativereason'])]
    y_tag=df_tag['negativereason']
    y_tag=y_tag.values
    X=df_tag['text'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y_tag, test_size=0.2, random_state=42)
    DT_param_grid = {
        'clf__estimator__criterion':['gini', 'entropy']
        ,'clf__estimator__splitter':['best','random']
        ,'clf__estimator__max_depth':[1,2,4,8,None]
        ,'clf__estimator__min_samples_split':[2,4,8]
        #,'clf_min_samples_leaf':[1,2,4,8]
        # ,'min_weight_fraction_leaf':[0.0
        # ,'max_features':[None
        # ,'random_state':[0
        # ,'max_leaf_nodes':[None
        # ,'class_weight':[None
        # ,'presort':[False
    }
    
    LinearSVC_grid_params={
        'clf__C':[0.01, 0.1, 0.5, 1]
        ,'clf__class_weight': ['balanced']
        #, 'dual"[True'
        #, 'fit_intercept':[True,
        #, 'intercept_scaling': [1,
        , 'clf__loss':['squared_hinge', 'hinge']
        , 'clf__max_iter':[250, 500, 1000, 2000]
        #, 'multi_class:['ovr'
        #, 'penalty':['l2'
        ,'clf__random_state':[42]
        #, 'tol':[0.0001,
        #verbose=0
    }
    
    
    #OneVsAll_DT: {'clf__estimator__max_depth': 8, 'clf__estimator__min_samples_split': 2, 'clf__estimator__criterion': 'entropy', 'clf__estimator__splitter': 'random'}

    
    estimator = build_tagger_pipe()
    #print(estimator.get_params().keys())
    #return None
    searcher = GridSearchCV(estimator=estimator, param_grid=LinearSVC_grid_params, cv=5, verbose=2, n_jobs=5)
    searcher.fit(X_train, y_train)
    print("Best parameters set found on development set:")
    print()
    print(searcher.best_params_)
    print()
    #print("Grid scores on development set:")
    #print()
    #for params, mean_score, scores in searcher.grid_scores_:
    #    print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
    #print()
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, searcher.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

    
    
if __name__ == '__main__':
    zero='1'
    learn='0' == zero
    stops = set(stopwords.words("english"))    
    if learn:
        do_learn()
    else:
        df = pd.read_csv('../input/Tweets.csv')

        #p=df[df['text'].str.contains('0162389030167')]
        #print([s for s in [t for t in p[:2]['text'].str.split('\s+')]])

        #uncomment if you dont care about neutral tweets
        #df = df.loc[df['airline_sentiment'].isin(['negative', 'positive'])]



        # y_sentiment=df['airline_sentiment']

        # import sys
        # sys.exit()


        t0 = time()
        #X_train = X_train[:3]
        #y_train = y_train[:3]


        # train_sentiment_model = True
        train_sentiment_model = False
        # train_tagger_model = False
        train_tagger_model = True

        if train_sentiment_model:
            '''
            negative    9178
            neutral     3099
            positive    2363
            '''
            # choose valid values
            print(df['airline_sentiment'].value_counts())
            df_sent = df.loc[df['airline_sentiment']!='neutral']
            print('----------------------------')
            print(df_sent['airline_sentiment'].value_counts())
            print('----------------------------')
            # sys.exit()

            X=df_sent['text'].values

            # y_sentiment=df_sent['airline_sentiment'].fillna('__EMPTY_SENTIMENT__')
            y_sentiment=df_sent['airline_sentiment']
            # y_sentiment=y_sentiment.map({'negative':0,'neutral':1, 'positive':2})
            y_sentiment = y_sentiment.values
            # y_sentiment = MultiLabelBinarizer().fit_transform(y_sentiment)
            # print(y_sentiment)
            X_train, X_test, y_train, y_test = train_test_split(X, y_sentiment, test_size=0.3, random_state=42)
            sentiment_model = build_sentiment_pipe()

            ## train model
            sentiment_model.fit(X_train, y_train)
            print("Training done in %0.3fs" % (time() - t0))
            print()
            scores = cross_val_score(sentiment_model, X_test, y_test)
            score = scores.mean()
            print("Score:{}".format(score))
            #print_significant_features(pipeline=pipeline)
            plot_significant_features(pipeline=sentiment_model)
            print("Total done in %0.3fs" % (time() - t0))

        if train_tagger_model:
            # choose valid values
            print(df.shape)
            print('-------------------------')
            df_tag = df[pd.notnull(df['negativereason'])]
            print(df_tag.shape)
            print('-------------------------')

            y_tag=df_tag['negativereason']
            print(y_tag.value_counts())
            y_tag=y_tag.values
            print('-------------------------')
            # sys.exit()

            X=df_tag['text'].values
            X_train, X_test, y_train, y_test = train_test_split(X, y_tag, test_size=0.2, random_state=42)
            OneVsAll_DT_educated = OneVsRestClassifier(DecisionTreeClassifier(
                class_weight='balanced', 
                max_depth= 8,
                min_samples_split= 2,
                criterion='entropy',
                splitter='random'
            ))
            
            LinearSVC_educated_args = {'loss': 'squared_hinge', 'C': 0.1, 'class_weight': 'balanced', 'max_iter': 250, 'random_state': 42}
            LinearSVC_educated = svm.LinearSVC(**LinearSVC_educated_args)
            Bagged_LinearSVC_educated = BaggingClassifier(base_estimator=svm.LinearSVC(**LinearSVC_educated_args),n_jobs=-1, n_estimators=1000)
            Bagged_LinearSVC_educated = BaggingClassifier(base_estimator=svm.LinearSVC(**LinearSVC_educated_args),n_jobs=-1, n_estimators=1000)
            #tagger_model = build_tagger_pipe(override_estimator=OneVsAll_DT_educated)
            tagger_model = build_tagger_pipe(LinearSVC_educated)
            
            ## train model
            tagger_model.fit(X_train, y_train)
            print("Training done in %0.3fs" % (time() - t0))
            print()
            scores = cross_val_score(tagger_model, X_test, y_test)
            score = scores.mean()
            print("Score:{}".format(score))
            print_significant_features(pipeline=tagger_model)
            plot_significant_features(pipeline=tagger_model)
            print("Total done in %0.3fs" % (time() - t0))
#Boosted_LinearSVC_educated is best do Grid search on it