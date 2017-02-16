# Tweeter airline topics
import os, joblib, re, unicodedata
import html
import numpy as np
import pandas as pd
#sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from nltk.stem.snowball import SnowballStemmer, EnglishStemmer

# My Modules
import sys
sys.path.append('../../../twitter/code/')
from nozzle import Nozzle

class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stemmer = EnglishStemmer()
        self._RE_tweet_emojis = re.compile(u'[\U0001F600-\U0001F64F]', re.UNICODE)
        self._RE_links = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')# links url
        self._RE_tweet_accounts = re.compile('@\\w+')
        self._RE_tweet_hashtag = re.compile('#\\w+')
        self._RE_ACRONYMS = re.compile('[A-Z]{2,}')
        self._RE_DATES =re.compile('\d+/\d+/?\d*')
        self._RE_WHITESPACES =re.compile('\s+')

    def clean_text(self, tweet):
        ''' tweet: tweet
        '''
        tweet = self._RE_WHITESPACES.sub(' ', tweet)
        tweet = html.unescape(tweet)
        tweet = self._RE_tweet_emojis.sub(lambda m: unicodedata.name(m.group()).upper().replace(' ', ''), tweet)
        # print(tweet)
        __LINK__='URLLINKURL' #'LINK'
        tweet = self._RE_links.sub(__LINK__, tweet)
        __ACCOUNTMENTION__ = 'they'#'ACCOUNTMENTION'#'ORGANIZATION'
        tweet = self._RE_tweet_accounts.sub(__ACCOUNTMENTION__, tweet)
        __HASHTAG__='it'#'HASHTAG'#'HASHTAG'
        tweet = self._RE_tweet_hashtag.sub(__HASHTAG__, tweet)
        # __ACRONYMS__=''#'ACRONYMS'
        # tweet = self._RE_ACRONYMS.sub(__ACRONYMS__, tweet)
        __DATES__='datemention'
        tweet = self._RE_DATES.sub(__DATES__, tweet)
        
        # tweet = " ".join([self.stemmer.stem(token) for token in tweet.split()])
        
        return tweet
    
    def transform(self, X, y=None):
        ''' Sentences as text
        '''
        # print(len(X))
        out = [self.clean_text(x) for x in X]
        # print(len(out))
        return out

    def fit_transform(self, X, y=None):
        return self.transform(X, y)
    
    def fit(self, X, y=None):
        self.transform(X)
        return self

        
def get_sentiment_data(file_path, refresh_cached=False, for_training=True):
    df=None
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(file_path))))
    cached_data_filename = DATA_DIR+'/cached_sentiment_{}_data.pkl.z'.format('training' if for_training else 'testing')
    if not refresh_cached and os.path.exists(cached_data_filename):
        df = joblib.load(cached_data_filename)
    else:
        print("Reading {} data".format('training' if for_training else 'testing'))
        df = pd.read_csv(file_path, encoding='utf-8')
        joblib.dump(df, cached_data_filename, compress=3)
    
    X=df['text'].values
   
    y=df['airline_sentiment']
    # y=y.fillna('NOREASON')
    y=y.values
    
    return X,y
    

def get_tagged_data(file_path, refresh_cached=False, for_training=True):
    '''
        file_path: path to data in csv format
    '''
    df=None
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(file_path))))
    cached_data_filename = DATA_DIR+'/cached_tagged_{}_data.pkl.z'.format('training' if for_training else 'testing')
    if not refresh_cached and os.path.exists(cached_data_filename):
        df = joblib.load(cached_data_filename)
    else:
        print("Reading {} data".format('training' if for_training else 'testing'))
        df = pd.read_csv(file_path, encoding='utf-8')
        joblib.dump(df, cached_data_filename, compress=3)
    
    # print(df.shape)
    X=df['text'].values
    # X=df['text'].values[:50]
    # with open("debug.txt", "wb") as text_file:
        # text_file.write("{0}".format("\n".join(X)).encode("UTF-8"))
    
    y=df['negativereason']
    y=y.fillna('NOREASON')
    y=y.values
    
    return X,y

def build_lda_model():
    n_topics = 8
    best_params = {
        'tfidf__norm': 'l2'
        ,'tfidf__use_idf': True
        ,'vectorizer__max_df': 1.0
        ,'vectorizer__max_features': None
        ,'vectorizer__ngram_range': (1, 2)
    }
    lda_pipe = Pipeline((
         ('cleantxt', TextCleaner())
        ,('vectorizer', CountVectorizer())
        ,('tfidf', TfidfTransformer()) #TfidfVectorizer
        ,('estimator', LatentDirichletAllocation(n_topics=n_topics, max_iter=5,learning_method='batch', learning_offset=50.,random_state=0, n_jobs=-1))
    ))
    return lda_pipe.set_params(**best_params)

def build_sentiment_model():

    best_params = {
        'estimator__alpha': 0.0000002# reduce and the likelihood of the class increases
        ,'estimator__n_iter': 5
        ,'estimator__penalty': 'elasticnet'
        ,'estimator__random_state':42
        ,'tfidf__norm': 'l2'
        ,'tfidf__use_idf': True
        ,'vectorizer__max_df': 1.0
        ,'vectorizer__max_features': None
        ,'vectorizer__ngram_range': (1, 2)
    }
    
    pipe_sentiment_model = Pipeline((
         ('cleantxt', TextCleaner())
        ,('vectorizer', CountVectorizer())
        ,('tfidf', TfidfTransformer()) #TfidfVectorizer
        ,('estimator', SGDClassifier(class_weight='balanced', loss='log'))
    ))
    return pipe_sentiment_model.set_params(**best_params)

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx, " ".join([feature_names[i]
            for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

def save_model(model, model_name=None, params=None):
    if model_name:
        model_name="{}.model.pkl.z".format(model_name)
    else:
        model_name="{}.model.pkl.z".format(type(model).__name__)
    joblib.dump(model, model_name, compress=True);
    
def load_model(model_name):
    return joblib.load(model_name)

if __name__=="__main__":
    data_file_path = '../input/Tweets.csv'
    X, y = get_tagged_data(data_file_path)

    #with new docs update lda model and reestimate new docs
    lda_model=None
    refresh_model=True
    model_name='tweeter_topics'
    if refresh_model:
        # print('model = lda(new_docs=new_docs)')
        lda_model = build_lda_model()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        lda_model = lda_model.fit(X_train, y_train)
        
        # with open("debug2.txt", "wb") as text_file:
            # text_file.write("{0}".format("\n".join( lda_model.get_params()['dbg'].X )).encode("UTF-8"))

        
        model_name="model__{}_{}".format(type(lda_model).__name__, model_name)
        save_model(lda_model, model_name)
    else:
        lda_model = load_model('model__Pipeline_tweeter_topics.model.pkl.z')
    
    tf_feature_names = lda_model.get_params()['vectorizer'].get_feature_names()
    n_top_words=5
    print_top_words(lda_model.get_params()['estimator'], tf_feature_names, n_top_words)
