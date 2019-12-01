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

from load_data import load_data

def average_len(df):
    return df['content_length'].mean()

def combine_tag(tags):
    for i in range(len(tags)):
        if tags[i] in ['sex','dating']:
            tags[i]='daing'
        if tags[i] in ['triggerwarning','urgent']:
            tags[i]='urgent'
        if tags[i] in ['wholesome','mentalhealth']:
            tags[i]='mentalhealth'
    return tags

def splitDataByTag(confessions):
    '''
    read data from several csv files and group them by tags
    param path(str): the directory of data files
    return: dictionary:{tag:dataframe}
    non-tag confessions
    taged confessions
    '''
    
    print(confessions.head())
    #non_tag_index=pd.isnull(confessions.tags)
    #non_tag=confessions.loc[non_tag_index]
    confessions=confessions.dropna(subset=['tags'])
    confessions['tags']=confessions.tags.str.replace(' ','')
    confessions['tags']=confessions.tags.str.split(',')
    confessions['tags']=confessions.tags.apply(combine_tag)
    #get frequent tags
    tags=defaultdict(int)
    for tag in confessions['tags']:
        for t in tag:
            tags[t]+=1
    tags=[(num,tag) for tag,num in tags.items() if num>=20]
    tags=sorted(tags,reverse=True)
    tags=[x[1] for x in tags]
    #divide confessions by tags
    tag_division={}
    for tag in tags:
        bool_index=[tag in confessions.loc[i,'tags'] for i in confessions.index]
        tag_division[tag]=confessions.loc[bool_index]
    return tag_division





if __name__ == "__main__":
    files = glob.glob('*.xlsx')
    confessions=load_data(files)
    print('loaded files:', files    )
    confessions['content_length']=[len(str(x)) for x in confessions.content]
    print(average_len(confessions))
    # confessions['timestamp']=pd.to_datetime(confessions.timestamp)
    confessions['yr-week']=confessions.timestamp.dt.strftime('%y-%W')
    confessions['hour']=confessions.timestamp.dt.strftime('%H')
    confessions['yr-month']=confessions.timestamp.dt.strftime('%y-%m')
    #confessions['year']=confessions.timestamp.dt.strftime('%y')
    confessions['weekday']=confessions.timestamp.dt.strftime('%a')
    confessions['weekday']=pd.Categorical(confessions['weekday'],categories=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'],ordered=True)
    print(confessions.head())
    print(confessions.info())
    
    weekday_distri=confessions.groupby('weekday')['content'].count().sort_index()
    mental_ill_keywords=['suicide','depression','fool','useless','dumb','cri','failure','fail']

    fig, ax = plt.subplots()
    ax.plot(list(weekday_distri.index),weekday_distri.values)
    ax.set(xlabel='weekday', ylabel='# of confessions')
    ax.grid()
    fig.savefig("weekday.png")
    plt.show()


    fig,ax=plt.subplots()
    week_distri=confessions.groupby('yr-week')['content'].count()
    ax.plot(week_distri.index,week_distri.values)
    ax.set(xlabel='year-week', ylabel='# of confessions')
    X=week_distri.index
    simple_X=[X[i] for i in range(len(X)) if i%3==0]
    ax.xaxis.set_ticks(simple_X)
    ax.grid()
    fig.savefig('year-week.png')
    plt.show()

    fig,ax=plt.subplots()
    month_distri=confessions.groupby('yr-month')['content'].count()
    ax.plot(month_distri.index,month_distri.values)
    ax.set(xlabel='year-month', ylabel='# of confessions')
    X=month_distri.index
    simple_X=[X[i] for i in range(len(X)) if i%2==0]
    ax.xaxis.set_ticks(simple_X)
    ax.grid()
    fig.savefig('year-month.png')
    plt.show()

    hour_distri=confessions.groupby('hour')['content'].count().sort_index()
    fig,ax=plt.subplots()
    
    ax.plot(hour_distri.index,hour_distri.values)
    ax.set(xlabel='hour',ylabel='# of confessions')
    X=hour_distri.index
    simple_X=[X[i] for i in range(len(X)) if i%2==0]
    ax.xaxis.set_ticks(simple_X)
    ax.grid()
    fig.savefig('hour.png')
    #ax2.xticks(list(range(24)))
    plt.show()
    
    
    keywords_of_each_tag={}
    confessions_divide_by_tags=splitDataByTag(confessions)
    avg_len_for_each_tag=[(average_len(df),tag) for tag,df in confessions_divide_by_tags.items()]
    avg_len_for_each_tag.sort()
       

    plt.rcdefaults()
    fig, ax = plt.subplots()
    tags = [x[1] for x in avg_len_for_each_tag]
    y_pos = np.arange(len(tags))
    performance = [x[0] for x in avg_len_for_each_tag]
    ax.barh(y_pos, performance, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tags)
    ax.set_xlabel('text length')
    ax.set_ylabel('tag')
    ax.set_title('average content length of each tag')
    fig.savefig('length.png')
    plt.show()

    







    
