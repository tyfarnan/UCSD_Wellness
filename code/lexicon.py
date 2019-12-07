import time
from lexicon_utils import *

weekdays = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

def preprocess(df):
  '''
  preprocess confessions to add needed data

  :param df: confessions
  :type df: pandas DataFrame
  :return: DataFrame with more informations
  '''
  df['weekday']=confessions.index.weekday
  df['weekday']=confessions['weekday'].apply(lambda day: weekdays[day])
  df['weekday']=pd.Categorical(df['weekday'],categories=weekdays,ordered=True)

  extra_clean_txt_file = r"data/extra_clean_comments.txt"
  extra_clean_csv_file = r"data/extra_clean_comments.csv"

  start = time.time()
  clean_text(confessions['content'], extra_clean_txt_file)
  clean_time = time.time() - start
  # print('Total time: ' + str(clean_time) + ' secs')

  clean_confessions = txt_to_csv(extra_clean_txt_file, extra_clean_csv_file, header=None)
  clean_content = clean_confessions[0].tolist()
