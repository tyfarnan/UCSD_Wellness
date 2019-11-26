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
        if tags[i] in ['rant','complaint']:
            tags[i]='complaint'
    return tags
    
def tag_num_category(tags):
    n=len(tags)
    if n>2:
        return '>=3'
    else:
        return str(n)

def plot_tag(confessions):

    one_tag_df=confessions.loc[confessions.tag_nums=='1']
    one_tag_df['tags']=[one_tag_df.loc[i,'tags'][0] for i in one_tag_df.index]
    one_tag_count=one_tag_df.groupby('tags')['content'].count()

    rare_tag_total_num=one_tag_count.loc[one_tag_count<=50].sum()
    one_tag_count=one_tag_count.loc[one_tag_count>50]
    one_tag_count.sort_values(ascending=False)
    print(one_tag_count)
    tag_num_count=confessions.groupby('tag_nums')['content'].count()

    # make figure and assign axis objects
    fig = plt.figure(figsize=(11, 6.0625))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    fig.subplots_adjust(wspace=0)
    # pie chart parameters
    ratios = [tag_num_count.loc['1']/tag_num_count.sum(), tag_num_count.loc['2']/tag_num_count.sum(), tag_num_count.loc['>=3']/tag_num_count.sum()]
    labels = ['one tag', 'two tags', 'three or more tags']
    explode = [0.1, 0, 0]
    # rotate so that first wedge is split by the x-axis
    angle = -180 * ratios[0]
    ax1.pie(ratios, autopct='%1.1f%%', startangle=angle,
        labels=labels, explode=explode)

    # bar chart parameters
    xpos = 0
    bottom = 0
    ratios = [one_tag_count[i]/(one_tag_count.sum()+rare_tag_total_num) for i in one_tag_count.index]
    ratios.append(rare_tag_total_num/(one_tag_count.sum()+rare_tag_total_num))
    width = .2
    colors = [[.1,.2,.3],[.1, .3, .3], [.1, .4, .3], [.1, .5, .5], [.1, .6, .7],[.1, .7, .9],[.1, .8, .7],[.1, .9, .5]]

    for j in range(len(ratios)):
        height = ratios[j]
        ax2.bar(xpos, height, width, bottom=bottom, color=colors[j])
        ypos = bottom + ax2.patches[j].get_height() / 2
        bottom += height
        ax2.text(xpos, ypos, "%d%%" % (ax2.patches[j].get_height() * 100),
             ha='center')

    ax2.set_title('tags distribution')
    ax2.legend(one_tag_count.index)
    ax2.axis('off')
    ax2.set_xlim(- 2.5 * width, 2.5 * width)

    # use ConnectionPatch to draw lines between the two plots
    # get the wedge data
    theta1, theta2 = ax1.patches[0].theta1, ax1.patches[0].theta2
    center, r = ax1.patches[0].center, ax1.patches[0].r
    bar_height = sum([item.get_height() for item in ax2.patches])

    # draw top connecting line
    x = r * np.cos(np.pi / 180 * theta2) + center[0]
    y = np.sin(np.pi / 180 * theta2) + center[1]
    con = ConnectionPatch(xyA=(- width / 2, bar_height), xyB=(x, y),
                      coordsA="data", coordsB="data", axesA=ax2, axesB=ax1)
    con.set_color([0, 0, 0])
    con.set_linewidth(4)
    ax2.add_artist(con)

    # draw bottom connecting line
    x = r * np.cos(np.pi / 180 * theta1) + center[0]
    y = np.sin(np.pi / 180 * theta1) + center[1]
    con = ConnectionPatch(xyA=(- width / 2, 0), xyB=(x, y), coordsA="data",
                      coordsB="data", axesA=ax2, axesB=ax1)
    con.set_color([0, 0, 0])
    ax2.add_artist(con)
    con.set_linewidth(4)

    plt.show()
    fig.savefig('tag_distribution_single.png')


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
    tags=[x[1] for x in tags]
    #divide confessions by tags
    tag_division={}
    for tag in tags:
        bool_index=[tag in confessions.loc[i,'tags'] for i in confessions.index]
        tag_division[tag]=confessions.loc[bool_index]
    return tag_division


def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%".format(pct, absolute)


if __name__ == "__main__":
    path='data'
    confessions=loadData(path)
    confessions['content_length']=[len(str(x)) for x in confessions.content]
    #time_process
    confessions['timestamp']=pd.to_datetime(confessions.timestamp)
    confessions['yr-week']=confessions.timestamp.dt.strftime('%y-%W')
    confessions['hour']=confessions.timestamp.dt.strftime('%H')
    confessions['yr-month']=confessions.timestamp.dt.strftime('%y-%m')
    confessions['weekday']=confessions.timestamp.dt.strftime('%a')
    confessions['weekday']=pd.Categorical(confessions['weekday'],categories=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'],ordered=True)

    #tag_process
    confessions=confessions.dropna(subset=['tags'])
    confessions['tags']=confessions.tags.str.replace(' ','')
    confessions['tags']=confessions.tags.str.split(',')
    confessions['tags']=confessions.tags.apply(combine_tag)
    confessions['tag_nums']=confessions.tags.apply(tag_num_category)

    #tag number plot
    tag_num=confessions.groupby('tag_nums')['content'].count()
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
   
    
    confessions_divide_by_tags=splitDataByTag(confessions)
    avg_len_for_each_tag=[(average_len(df),tag) for tag,df in confessions_divide_by_tags.items()]
    avg_len_for_each_tag.sort()

       
    #length-tag
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
    fig.savefig('tag_length.png')
    plot_tag(confessions)

    #tag frequency
    tag_alone_nums=[(df.loc[df.tag_nums=='1'].shape[0],tag) for tag,df in confessions_divide_by_tags.items()]
    tag_multi_nums=[(df.loc[df.tag_nums!='1'].shape[0],tag) for tag,df in confessions_divide_by_tags.items()]
    print(tag_alone_nums)
    print(tag_multi_nums)

    N = len(tag_alone_nums)
    alone = [num[0] for num in tag_alone_nums]
    multi = [num[0] for num in tag_multi_nums]
    ind = [num[1] for num in tag_alone_nums]    # the x locations for the groups
    width = 0.35       # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(ind, multi, width)
    p2 = plt.bar(ind, alone, width,
             bottom=multi)

    plt.ylabel('frequency')
    plt.title('')
    plt.legend((p1[0], p2[0]), ('multiple tags', 'single tag'))
    plt.show()
    








    
