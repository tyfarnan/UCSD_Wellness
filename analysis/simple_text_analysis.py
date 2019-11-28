#Data Analysis
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

#Data Preprocessing and Feature Engineering
from textblob import TextBlob
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#Model Selection and Validation
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

#functions in tag analysis
from tag_analysis import loadData
from collections import defaultdict
import math

#extract useful information
def form_sentence(tweet):
    tweet_blob = TextBlob(str(tweet))
    return ' '.join(tweet_blob.words)

#text clean
def no_number_alpha(tweet):
    tweet_list = tweet.split()
    clean_tokens = [t for t in tweet_list if re.match(r'[^\W\d]*$', t)]
    clean_s = ' '.join(clean_tokens)
    clean_mess = [word.lower() for word in clean_s.split() if word.lower() not in stopwords.words('english')]
    return clean_mess
#word normalization
def normalization(tweet_list):
    lem = WordNetLemmatizer()
    normalized_tweet = []
    for word in tweet_list:
        normalized_text = lem.lemmatize(word,'v')
        normalized_tweet.append(normalized_text)
    return normalized_tweet

def tf_idf_one(words):
    word_count_one_doc=defaultdict(int)
    for w in words:
        word_count_one_doc[w]+=1
    idf={w:math.log(N/all_doc_word_count[w],10) for w in word_count_one_doc}
    tfidf={w:word_count_one_doc[w]*idf[w] for w in word_count_one_doc}
    tfidf_count=[(tfidf[w],w) for w in tfidf]
    tfidf_count.sort(reverse=True)
    return [w_c[1] for w_c in tfidf_count[:5]]

def contain_words(words):
    for w in words:
        if w in mental_ill_keywords:
            return 1
    return 0

if __name__ == "__main__":
    path='data'
    confessions=loadData(path)
    print(confessions.head())
    print(confessions.info())
    confessions['content_processed']=confessions.content.apply(form_sentence)
    confessions['content_processed']=confessions.content_processed.apply(no_number_alpha)
    confessions['content_processed']=confessions.content_processed.apply(normalization)
    '''
    #word_count
    all_doc_word_count=defaultdict(int)
    N=0
    for i in confessions.index:
        for w in confessions.loc[i,'content_processed']:
            N+=1
            all_doc_word_count[w]+=1
'''
   
    mental_ill_keywords=set(['suicide','depression','fool','useless','dumb','cry','failure','fail','depress'])
    confessions['contain_men_ill_w']=confessions.content_processed.apply(contain_words)
    print(confessions.contain_men_ill_w.sum())