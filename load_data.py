import collections
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from collections import defaultdict
import statistics as stats

def load_data(files=None, index_time=False):
  '''
  loads confessions data from all sheets in excel files and convert to
  pandas dataframe indexed by timestamp

  :param files: name/path to the excel files to load
  :type files: list of str
  '''

  if files == None:
  # by default load files in current directory
    import glob
    files = glob.glob('./*.xlsx')

  assert isinstance(files, list)
  # iterate all files and get sheets
  sheets = list()
  for file in files:
    assert isinstance(file, str)
    read = pd.read_excel(file, sheet_name=None)
    if isinstance(read, collections.OrderedDict):
      # file has more than one sheet, add em all
      for sheet in read:
        sheets.append(read[sheet])
    else:
      # files has one sheet, add it
      sheets.append(read)

  # contatenate sheets
  df = pd.concat(sheets, join='outer', sort=False)
  # convert timestamp strings to pandas timestamp
  df['timestamp'] = pd.to_datetime(df['timestamp'])

  if index_time:
    # set the timestamp as index and sort
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    df.index.name = 'timestamp'
  else:
    # otherwise just sort by timestamp column
    df.sort_values(by=['timestamp'], ascending=True, inplace=True)

  return df

def split_tags(confessions):
    '''
    read data from the confessions dataframe and create groups by tags

    :param confessions: pandas DataFrame of confessions
    :return: dictionary {tag : dataframe}

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

def average_len(df):
    '''
    calculate average length of content column (in characters)

    :param confessions: pandas DataFrame of confessions
    :return: the average
    '''
    return stats.mean( map( lambda x: len(str(x)), df.content ) )

def combine_tag(tags):
    '''
    arbitrary function
    '''
    for i in range(len(tags)):
        if tags[i] in ['sex','dating']:
            tags[i]='dating'
        if tags[i] in ['serious','urgent']:
            tags[i]='serious'
        # if tags[i] in ['triggerwarning','urgent']:
        #     tags[i]='urgent'
        # if tags[i] in ['wholesome','mentalhealth']:
        #     tags[i]='mentalhealth'
    return tags

if __name__ == "__main__":
  print('Loading Excel spreadsheets...')
  files = ["DatasetMain_use.xlsx", "DatasetOlder_use.xlsx"]
  df = load_data(files)
  print(df)