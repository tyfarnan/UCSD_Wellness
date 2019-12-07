import pickle
import collections
import glob
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from collections import defaultdict
import statistics as stats
import json
import datetime as dt

# default folder and files
dfolder = 'data'
default = {
  'xlsx': dfolder+'/*.xlsx',
  'corpus': dfolder+'/*_corpus.txt',
  'calendar': dfolder+'/academic_calendar.json',
  'cache': 'cache.dat'
}

def load_data(files=None, index_time=True):
  '''
  loads confessions data from all sheets in excel files and convert to
  pandas dataframe indexed by timestamp

  if no files are specified, will load all files in default directory

  :param files: name/path to the excel files to load
  :type files: list of str
  :return: combined confessions dataframe
  '''

  if files == None:
  # by default load files in default directory
    files = glob.glob(default['xlsx'])

  assert isinstance(files, list)
  # iterate all files and get sheets
  sheets = list()
  for file in files:
    assert isinstance(file, str)
    try:
      read = pd.read_excel(file, sheet_name=None)
    except:
      read = pd.read_csv(file)
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

def load_calendar(file=default['calendar']):
  '''
  loads academic calendar from json file

  :param file: name/path to the json file to load
  :type file: str
  :return: calendar dict
  '''
  assert isinstance(file, str)
  # https://realpython.com/python-json/
  with open(file, "r") as read_file:
    data = json.load(read_file)
  assert 'quarters' in data
  quarters= data['quarters']
  for quarter in quarters:
    for key in quarter:
      try:
        quarter[key] = dt.datetime.strptime(quarter[key], "%Y-%m-%d")
      except:
        pass
  return quarters

def load_cache(cache_file='cache.dat'):
  '''
  cache wrapper excel and json files for faster startup

  :param cache_file: name/path to the cache file to use
  :type cache_file: str
  :return: confessions dataframe, calendar dict
  '''
  assert isinstance(cache_file, str)
  try:
    df, calendar = pickle.load( open( cache_file, "rb" ) )
    print("Loaded previous cached data...")
  except FileNotFoundError:
    print("Previous data not found.")
    df = load_data()
    calendar = load_calendar()
    pickle.dump( (df, calendar), open( cache_file, "wb" ) )
    print('Loaded data...')
  return df, calendar

def load_corpus(file):
  '''
  loads a corpus of words from file

  :param file: file name of the corpus
  :type file: str
  :return: words in list format
  '''
  assert isinstance(file, str)
  with open(file, 'r') as f:
    words = [line.strip('\n') for line in f]
  return words

def split_tags(confessions):
    '''
    read data from the confessions dataframe and create groups by tags

    :param confessions: set of confessions
    :type confessions: pandas DataFrame
    :return: dictionary {tag : dataframe}
    '''
    assert isinstance(confessions, pd.DataFrame)
    print(confessions.head())
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

    :param confessions: set of confessions
    :type confessions: pandas DataFrame
    :return: the average length in float
    '''
    assert isinstance(df, pd.DataFrame)
    return stats.mean( map( lambda x: len(str(x)), df.content ) )

if __name__ == "__main__":
  # test all functions

  print('Loading Excel spreadsheets...')

  # df = load_data()
  # calendar = load_calendar()
  df, calendar = data_cache()

  print(df, '\n\n')
  print(average_len(df), '\n\n')
  print(calendar, '\n\n')

  corpuses = glob.glob(default['corpus'])
  for filename in corpuses:
    corpus = load_corpus(filename)
    print(corpus, '\n\n')
