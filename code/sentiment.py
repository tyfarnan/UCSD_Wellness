#Data Analysis
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import math
import pandas as pd 
import time
import glob
from collections import defaultdict,Counter
from nltk.stem.porter import *
import string
import numpy as np
from nltk.corpus import stopwords
import math
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

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


from collections import defaultdict
import math
import pandas as pd 
import time
import glob
from collections import defaultdict,Counter
from nltk.stem.porter import *
import string
import numpy as np
from nltk.corpus import stopwords
import math
from matplotlib import pyplot as plt

weekdays = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

#extract useful information
def loadData(path='data'):
    '''
    read data from several csv files
    param path(str): the directory of data files
    return: dataframe
    '''
    assert isinstance(path,str)
    files = glob.glob(path+'/*.csv')
    df=pd.concat([pd.read_csv(f) for f in files])
    return df

def form_sentence(tweet):
    '''
    extract useful information from text
    '''
    assert isinstance(tweet,object)
    tweet_blob = TextBlob(str(tweet))
    return ' '.join(tweet_blob.words)

#text clean
def no_number_alpha(tweet):
    '''
    remove numbers and strange characters from text and generate word list
    '''
    assert isinstance(tweet,str)
    tweet_list = tweet.split()
    clean_tokens = [t for t in tweet_list if re.match(r'[^\W\d]*$', t)]
    clean_s = ' '.join(clean_tokens)
    clean_mess = [word.lower() for word in clean_s.split() if word.lower() not in stopwords.words('english')]
    return clean_mess
#word normalization
def normalization(tweet):
    '''
    normalize word into simplified mode
    '''
    assert isinstance(tweet,object)
    tweet=form_sentence(tweet)
    tweet_list=no_number_alpha(tweet)
    lem = WordNetLemmatizer()
    normalized_tweet = []
    for word in tweet_list:
        normalized_text = lem.lemmatize(word,'v')
        normalized_tweet.append(normalized_text)
    return normalized_tweet

def load_sentiment_dict(path='data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'):
    '''
    load lexicon file into a word:list of sentiments dictionary
    '''
    assert isinstance(path,str)
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

def sentiment_category(content_list,word_senti):
    '''
    sentiment classification for text
    param content_list: cleaned text words
    type content_list: list of str
    rtype: str
    '''
    assert isinstance(content_list,list)
    assert isinstance(word_senti,dict)
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
    

def two_class_sentiment_category(content_list,word_senti):
    '''
    basic sentiment classification for text
    param content_list: cleaned text words
    type content_list: list of str
    rtype: str
    '''
    assert isinstance(content_list,list)
    assert isinstance(word_senti,dict)
    sentiment_score=defaultdict(int)
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
    
def hour_plot(confessions):
    '''
    plot confessions count in 24 hours
    type confessions: dataframe
    '''
    assert isinstance(confessions,pd.DataFrame)
    hour_distri=confessions.groupby('hour')['content'].count().sort_index()
    fig,ax=plt.subplots()
    ax.plot(hour_distri.index,hour_distri.values)
    ax.set(xlabel='Hour',ylabel='Count',title='Confessions Count by Hour')
    X=hour_distri.index
    simple_X=[X[i] for i in range(len(X)) if i%4==0]
    simple_X.append('24')
    ax.xaxis.set_ticks(simple_X)
    #ax.grid()
    #fig.savefig('hour.png')
    plt.show()
    
def weekday_plot(confessions, file=None):
    '''
    plot confessions count in each weekday
    type confessions: dataframe
    '''
    assert isinstance(confessions,pd.DataFrame)
    weekday_distri=confessions.groupby('weekday')['content'].count().sort_index()
    fig, ax = plt.subplots()
    ax.plot(list(weekday_distri.index),weekday_distri.values)
    ax.set(xlabel='Weekday', ylabel='Count',title='Confessions Count by Weekday')
    #ax.grid()
    # fig.savefig("weekday.png")
    if file: plt.savefig(file)
    plt.show()

def week_plot(confessions, file=None):
    '''
    plot confessions count in each week
    type confessions: dataframe
    '''
    assert isinstance(confessions,pd.DataFrame)
    fig,ax=plt.subplots()
    week_distri=confessions.groupby('yr-week')['content'].count()
    ax.plot(week_distri.index,week_distri.values)
    ax.set(xlabel='Year-Week', ylabel='Count',title='Confessions Count by Week')
    X=week_distri.index
    simple_X=[X[i] for i in range(len(X)) if i%5==0]
    ax.xaxis.set_ticks(simple_X)
    #ax.grid()
    # fig.savefig('year-week.png')
    if file: plt.savefig(file)
    plt.show()

def month_plot(confessions, file=None):
    '''
    plot confessions count in each month
    type confessions: dataframe
    '''
    assert isinstance(confessions,pd.DataFrame)
    fig,ax=plt.subplots()
    month_distri=confessions.groupby('yr-month')['content'].count().sort_index()
    ax.plot(list(month_distri.index) ,month_distri.values)
    ax.set(xlabel='Month', ylabel='Count',title='Confessions Count by Month')
    #X=month_distri.index
    #simple_X=[X[i] for i in range(len(X)) if i%2==0]
    #ax.xaxis.set_ticks(simple_X)
    #ax.grid()
    # fig.savefig('year-month.png')
    if file: plt.savefig(file)
    plt.show()

def pre_process(sentiments_of_word, confessions):
    '''
    load data and add caracter columns to dataframe
    '''
    assert isinstance(confessions, pd.DataFrame)
    confessions=confessions.loc[:,['timestamp','content']]
    confessions['content_processed']=confessions.content.apply(normalization)
    confessions['sentiment']=confessions.content_processed.apply(sentiment_category,word_senti=sentiments_of_word)
    confessions['pos_neg']=confessions.content_processed.apply(two_class_sentiment_category,word_senti=sentiments_of_word)
    confessions['timestamp']=confessions.index
    confessions['ts']=confessions.timestamp.values.astype(np.int64)
    confessions['yr-week']=confessions.timestamp.dt.strftime('%y-%W')
    confessions['hour']=confessions.timestamp.dt.strftime('%H')
    confessions['yr-month']=confessions.timestamp.dt.strftime('%b')
    confessions['yr-month']=pd.Categorical(confessions['yr-month'],categories=['Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun','Juy','Aug','Sep'],ordered=True)
    confessions['weekday']=confessions.timestamp.dt.strftime('%a')
    confessions['weekday']=pd.Categorical(confessions['weekday'],categories=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'],ordered=True)
    #confessions['quarter']=confessions.ts.apply(split_quarter)
    confessions['date']=confessions.timestamp.dt.strftime('%m-%d')
    return confessions

def school_calendar_timestamp():
    '''
    generate a dataframe containing the start and end timestamp of each quarter
    '''
    df = pd.DataFrame({'year': [2018, 2018,2019,2019,2019,2019],
                                 'month': [9, 12 ,1, 3 ,5 ,6],
                                  'day': [24, 15, 2, 23, 27, 14]})
    s=pd.to_datetime(df)
    df=s.to_frame(name='datetime')
    df['ts'] = df.datetime.values.astype(np.int64)
    quarter_ts=[df.loc[i,'ts']  for i in df.index]
    label=['fall_begin','fall_end','winter_begin','winter_end','spring_begin','spring_end']
    school_calendar=dict(zip(label,quarter_ts))
    return school_calendar

def split_quarter(timestamp):
    '''
    add quarter labels to each confession
    '''
    assert isinstance(timestamp,int)
    if timestamp<school_calendar['fall_end'] and timestamp>school_calendar['fall_begin']:
        return 'Fall'
    if timestamp<school_calendar['winter_end'] and timestamp>school_calendar['winter_begin']:
        return 'Winter'
    if timestamp<school_calendar['spring_end'] and timestamp>school_calendar['spring_begin']:
        return 'Spring'

    
    
def sentiment_plot(confessions, file=None):
    '''
    plot confessions count of each sentiment
    '''
    assert isinstance(confessions,pd.DataFrame)
    senti_category_count=confessions.groupby('sentiment')['content'].count().sort_values(ascending=False)
    width=0.35
    plt.figure(figsize=(12,4))
    plt.bar(senti_category_count.index,senti_category_count.values,width,color=['tab:red','tab:blue','tab:red','tab:red','tab:blue','tab:blue','tab:red','tab:blue'])
    plt.ylabel('Count')
    plt.xlabel('Emotions')
    plt.title('Confessions Count by Emotions')
    red_patch = mpatches.Patch(color='tab:red', label='Negative')
    blue_patch = mpatches.Patch(color='tab:blue', label='Positive')
    plt.legend(handles=[red_patch,blue_patch],prop={'size':15})
    if file: plt.savefig(file)
    plt.show()

def neg_pos_plot(confessions, file=None):
    '''
    plot confessions count of each basic sentiment
    '''
    assert isinstance(confessions,pd.DataFrame)
    pos_neg_category_count=confessions.groupby('pos_neg')['content'].count()
    #width=0.35
    plt.bar(pos_neg_category_count.index,pos_neg_category_count.values,color=['tab:red','tab:blue'])
    plt.ylabel('Count')
    plt.xlabel('Basic Emotions')
    plt.title('Confessions Count by Basic Emotions')
    
    if file: plt.savefig(file)
    plt.show()

def normalize_series(s):
    '''
    normalize data
    '''
    assert isinstance(s,pd.Series)
    total_num=s.sum()
    s=s/total_num
    return s
def weekday_diff_senti_plot(confessions, file=None):
    '''
    plot the trend of three emotions'count in each weekday 
    '''
    assert isinstance(confessions,pd.DataFrame)
    # confessions['week_d']=confessions.index.strftime('%a')
    confessions['week_d']=confessions.index.weekday
    confessions['week_d']=confessions['week_d'].apply(lambda day: weekdays[day])
    confessions['week_d']=pd.Categorical(confessions['week_d'],categories=weekdays[:-2],ordered=True)
    joy=confessions.loc[confessions.sentiment=='joy'].groupby('week_d')['content'].count().sort_index()
    fear=confessions.loc[confessions.sentiment=='fear'].groupby('week_d')['content'].count().sort_index()
    anger=confessions.loc[confessions.sentiment=='anger'].groupby('week_d')['content'].count().sort_index()
    joy=normalize_series(joy)
    fear=normalize_series(fear)
    anger=normalize_series(anger)
    p1=plt.plot(list(joy.index),joy.values,label='joy',color='b')
    p2=plt.plot(list(fear.index),fear.values,label='fear',color='y')
    p3=plt.plot(list(anger.index),anger.values,label='anger',color='r')
    plt.xlabel('Weekday')
    # plt.xticks(['Mon','Tue','Wed','Thu','Fri'])
    plt.ylabel('Count (Normalized)')
    plt.title('Confessions Count of Different Emotions by Weekday')
    #plt.grid()
    plt.legend(title='Emotions')
    if file: plt.savefig(file)
    plt.show()


if __name__ == "__main__":
    # test functionality
    from load_data import load_data
    df=load_data()
    school_calendar=school_calendar_timestamp()

    import pickle
    try:
        sentiments_of_word, confessions = pickle.load( open( 'sentiments_cache.dat', "rb" ) )
        print("Loaded previous cached data...")
    except FileNotFoundError:
        print("Previous data not found.")
        sentiments_of_word=load_sentiment_dict()
        confessions=pre_process(sentiments_of_word, df)
        pickle.dump(    (sentiments_of_word, confessions), 
                        open( 'sentiments_cache.dat', "wb" ) )
        print('Loaded data...')

    sentiment_plot(confessions)

    neg_pos_plot(confessions)
    weekday_diff_senti_plot(confessions)
   
    
