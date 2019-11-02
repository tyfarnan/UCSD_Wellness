import pandas as pd 
import time
from collections import defaultdict,Counter
path='confession_data.csv'
confessions=pd.read_csv(path)
#print(confessions.head())
print(confessions.info())
print(confessions.head())
confessions['tags']=confessions['tags'].str.replace(' ','')
confessions['tags']=confessions['tags'].str.split(',')
#confessions['timestamp']=map(time.strptime('%Y-%m-%d %H:%M:%S'),confessions['timestamp'])
time_struct=[]
time_stamp=[]
time_date=[]
time_year=[]
time_month=[]
for t_str in confessions['timestamp']:
    t=time.strptime(t_str,'%Y-%m-%d %H:%M:%S')
    t_stamp=time.mktime(t)
    date=time.strftime('%Y-%m-%d',t)
    year=time.strftime('%Y',t)
    month=time.strftime('%m',t)
    time_struct.append(t)
    time_date.append(date)
    time_stamp.append(t_stamp)
    time_month.append(month)
    time_year.append(year)
print(time_date[:10])
confessions['date']=time_date
confessions['year']=time_year
confessions['month']=time_month
confessions['time_struct']=time_struct
print(confessions.groupby(['month']).count())
print(confessions.groupby(['year']).count())
#print(confessions.head())
tags=[]
for tag in confessions['tags']:
    if isinstance(tag,list):
        for t in tag:
            tags.append(t)


c=Counter(tags)

for tag in c:
    #if c[tag]>10:
        print(tag+': '+str(c[tag]))
