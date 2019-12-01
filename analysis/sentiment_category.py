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
def normalization(tweet):
    tweet=form_sentence(tweet)
    tweet_list=no_number_alpha(tweet)
    lem = WordNetLemmatizer()
    normalized_tweet = []
    for word in tweet_list:
        normalized_text = lem.lemmatize(word,'v')
        normalized_tweet.append(normalized_text)
    return normalized_tweet

def load_sentiment_dict(path='/Users/zmhlala/UCSD_Wellness/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'):
    word_senti=open(path,'r')
    word_sentiment_dict=defaultdict(list)
    sentiment_set=set()
    for w_sent in word_senti.readlines():
        w_sent=w_sent.strip('\n').split('\t')
        if len(w_sent)<3:
            print(w_sent)
            continue
        word=w_sent[0]
        sentiment=w_sent[1]
        if w_sent[2]=='1':
            word_sentiment_dict[word].append(sentiment)
        sentiment_set.add(sentiment)
    word_senti.close()
    return word_sentiment_dict

def sentiment_category(content_list):
    
    sentiment_score=defaultdict(int)
    for w in content_list:
        if w in word_senti:
            sentiments=word_senti[w]
            for senti in sentiments:
                if senti not in ['positive','negative']:
                    sentiment_score[senti]+=1
    
    score_senti=[(score,senti) for senti,score in sentiment_score.items()]
    score_senti.sort()
    if score_senti:
        return score_senti[0][1]
    

def two_class_sentiment_category(content_list):
    
    sentiment_score=defaultdict(int)
    for w in content_list:
        if w in word_senti:
            sentiments=word_senti[w]
            for senti in sentiments:
                if senti in ['positive','negative']:
                    sentiment_score[senti]+=1
    
    score_senti=[(score,senti) for senti,score in sentiment_score.items()]
    score_senti.sort()
    if score_senti:
        return score_senti[0][1]
    
    


if __name__ == "__main__":
    path='data'
    confessions=loadData(path)
    word_senti=load_sentiment_dict()
    confessions['content_processed']=confessions.content.apply(normalization)
    confessions['sentiment']=confessions.content_processed.apply(sentiment_category)
    confessions['pos_neg']=confessions.content_processed.apply(two_class_sentiment_category)

    senti_category_count=confessions.groupby('sentiment')['content'].count()
    pos_neg_category_count=confessions.groupby('pos_neg')['content'].count()

    print(senti_category_count)
    print(pos_neg_category_count)
    
    
