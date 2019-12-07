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
from matplotlib.patches import ConnectionPatch

def tag_num_category(tags):
    '''
    category confessions by how many tags they have 
    '''
    assert not tags or isinstance(tags,list)
    if tags:
        n=len(tags)
        if n>2:
            return '>=3'
        else:
            return str(n)
    else:
        return '0'

def tag_process(tags):
    '''
    split tag string into a list
    '''
    assert pd.isnull(tags) or isinstance(tags,str)
    if not pd.isnull(tags):
        tags=tags.replace(' ','').split(',')
        return tags
    
def splitDataByTag(confessions):
    '''
    read data from several csv files and group them by tags
    param path(str): the directory of data files
    return: dictionary:{tag:dataframe}
    non-tag confessions
    taged confessions
    '''
    
    #get frequent tags
    assert isinstance(confessions,pd.DataFrame)
    tags=defaultdict(int)
    for tag in confessions['tags']:
        for t in tag:
            tags[t]+=1
    tags=[(num,tag) for tag,num in tags.items() if num>=50]
    tags=sorted(tags,reverse=True)
    return tags

def preprocess_data(confessions):
    '''
    load data, add 'tag' and 'tag_num' columns to dataframe
    '''
    assert isinstance(confessions, pd.DataFrame)
    confessions['tags']=confessions.tags.apply(tag_process)
    confessions['tag_num']=confessions.tags.apply(tag_num_category)
    return confessions

def func(pct, allvals):
    '''
    count the ratio of each part
    '''
    assert isinstance(allvals,np.ndarray)
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%".format(pct, absolute)

def tag_num_plot(confessions, file=None):
    '''
    plot confessions count by tag_num
    '''
    assert isinstance(confessions,pd.DataFrame)
    tag_num=confessions.groupby('tag_num')['content'].count()
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
    data=tag_num.values
    numbers=tag_num.index
    
    wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),
                                  textprops=dict(color="w"),colors=['r','m','g','y'])
    ax.legend(wedges, numbers,
          title="Tag Numbers",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))
    plt.setp(autotexts, size=12, weight="bold")
    plt.title('Confessions Count by Tag Numbers')
    if file: plt.savefig(file)
    plt.show()

def tag_plot(confessions, file=None):
    '''
    plot confessions count by tag
    '''
    assert isinstance(confessions,pd.DataFrame)
    confessions_tagged=confessions.dropna(subset=['tags'])
    tags_num=splitDataByTag(confessions_tagged)
    width=0.35
    tags=[d[1] for d in tags_num]
    nums=[d[0] for d in tags_num]
    plt.figure(figsize=(14,4))
    plt.bar(tags,nums,width)
    plt.xlabel('Tags')
    plt.ylabel('Count')
    plt.title('Confessions Count by Tags')
    if file: plt.savefig(file)
    plt.show()

if __name__ == "__main__":
    # test functionality
    from load_data import load_data
    df=load_data()
    confessions=preprocess_data(df)
    #tag number plot
    tag_num_plot(confessions)
    #tag frequency
    tag_plot(confessions)
