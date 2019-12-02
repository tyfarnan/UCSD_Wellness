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

def loadData(path):
    '''
    read data from several csv files
    param path(str): the directory of data files
    return: dataframe
    '''
    files = glob.glob(path+'/*.csv')
    df=pd.concat([pd.read_csv(f) for f in files])
    return df

def tag_num_category(tags):
    if tags:
        n=len(tags)
        if n>2:
            return '>=3'
        else:
            return str(n)
    else:
        return '0'

def tag_process(tags):
    if not pd.isnull(tags):
        tags=tags.replace(' ','').split(',')
        return tags


def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%".format(pct, absolute)

def splitDataByTag(confessions):
    '''
    read data from several csv files and group them by tags
    param path(str): the directory of data files
    return: dictionary:{tag:dataframe}
    non-tag confessions
    taged confessions
    '''
    
    #get frequent tags
    tags=defaultdict(int)
    for tag in confessions['tags']:
        for t in tag:
            tags[t]+=1
    tags=[(num,tag) for tag,num in tags.items() if num>=50]
    tags=sorted(tags,reverse=True)
    return tags

if __name__ == "__main__":
    path='data'
    confessions=loadData(path)
    confessions['tags']=confessions.tags.apply(tag_process)
    confessions['tag_num']=confessions.tags.apply(tag_num_category)
    print(confessions.head())

    #tag number plot
    tag_num=confessions.groupby('tag_num')['content'].count()
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
    data=tag_num.values
    numbers=tag_num.index
    
    wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),
                                  textprops=dict(color="w"))
    ax.legend(wedges, numbers,
          title="Number of tags",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))
    plt.setp(autotexts, size=12, weight="bold")
    fig.savefig('tag_num_distribution.png')
    plt.show()

    #tag frequency
    confessions_tagged=confessions.dropna(subset=['tags'])
    tags_num=splitDataByTag(confessions_tagged)
    width=0.35
    tags=[d[1] for d in tags_num]
    nums=[d[0] for d in tags_num]
    plt.bar(tags,nums,width)
    plt.xlabel('tags')
    plt.ylabel('# of confessions')
    plt.show()