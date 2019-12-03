import glob
import pickle
from load_data import load_data
from load_data import load_calendar
from load_data import load_corpus
from distributions import add_extra_time_units
from distributions import histogram
from distributions import plot_histogram
from distributions import plot_custom
from distributions import plot_weekday_dist
from distributions import plot_hourly_dist
from distributions import submissions_over_time
from distributions import set_show_plots
from distributions import label_macro_size
from distributions import auto_reindex_lut
import datetime as dt
import pandas as pd
import seaborn as sns
import string

def load_cache(cache_file='cache.dat'):
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

def get_quarter_start_end(quarter):
  try: mask_begin = quarter['start'] # try official start
  except: mask_begin = quarter['begins'] # else use instruction start
  mask_end = quarter['end'] # get end of quarter (after finales)
  return (mask_begin, mask_end)

def hist_hourly_quarter_edges(df, calendar, span):
  quarter_spans = list(map(get_quarter_start_end, calendar))

  df_front_list = list(map(lambda s: df.loc[s[0]:s[0]+dt.timedelta(weeks=span)], quarter_spans))
  df_front = pd.concat(df_front_list)
  hist_front = histogram(df_front, 'hour')

  df_rear_list = list(map(lambda s: df.loc[s[1]-dt.timedelta(weeks=span):s[1]], quarter_spans))
  df_rear = pd.concat(df_rear_list)
  hist_rear = histogram(df_rear, 'hour')

  hist_comb = pd.concat([hist_front, hist_rear], axis=1)
  hist_comb.columns=['First '+str(span)+' Weeks', 'Last '+str(span)+' Weeks']
  return hist_comb

def hist_hourly_weekday_weeked(df):
  weekday_indexes = range(0, 4) # monday-thursday
  weekend_indexes = range(4, 7) # friday-sunday

  df_weekday = df[df.index.weekday < 4] # monday-thursday
  hist_weekday = histogram(df_weekday, 'hour')
  df_weekend = df[df.index.weekday >= 4] # friday-sunday
  hist_weekend = histogram(df_weekend, 'hour')

  hist_comb = pd.concat([hist_weekday, hist_weekend], axis=1)
  hist_comb.columns=['Mon-Thurs', 'Fri-Sun']
  return hist_comb

def count_words(words, content):
  filtered = list(map(lambda x: x.strip().translate(str.maketrans('', '', string.punctuation)), content.split()))
  return sum(map(lambda word: word in words, filtered))

import re
def keywords_over_quarters(corpus, calendar, df, title=None, file=None, colors=['lightgrey', 'tab:red', 'tab:green']):
  
  df_total = pd.DataFrame(columns=['content', 'words'])
  # challenge: merge multiple quarters together
  for quarter in calendar:
    qtr_slicer = slice(quarter['start'], quarter['end'])
    df_part = df.loc[ qtr_slicer ]
    df_part['qtr_day'] = (df_part.index.date - quarter['start'].date()) # get day of the quarter
    df_part['words'] = df_part['content'].apply( lambda x: count_words(corpus, str(x)) )
    dist_total = df_part.groupby(df_part.qtr_day).count().content
    dist_stress = df_part.groupby(df_part.qtr_day).sum().words
    dist = pd.concat([dist_total, dist_stress], axis=1)
    df_total=df_total.add(dist, fill_value=0)  # cumulatve add
  df_total = df_total.astype({'content':int,'words':int}, copy=False)
  df_total['week_num'] = df_total.index.map(lambda x: (x.days // 7))
  df_weekly = df_total.groupby(df_total.week_num).sum()
  # df_norm = (df_weekly-df_weekly.mean())/df_weekly.std()
  df_norm = ( df_weekly ) / ( df_weekly.max() - df_weekly.min() )

  df_norm.rename(columns={'content':'Submissions', 'words': 'Keywords'}, inplace=True)
  def custom_graph(ax):
    # ax.axhline(0, label='Average', color='dimgrey', linestyle='--')
    scale_bottom = 0.85*df_norm.min().min()
    scale_top = 1.15*df_norm.max().max()
    ax.set_ylim(bottom=scale_bottom, top=scale_top)
    ax.set_xticklabels(df_norm.index, rotation=0)
    return ax

  plot_custom(df_norm, 'bar', 
                extra_func=custom_graph,
                xlabel='Week of Quarter', ylabel='Activity Level',
                title=title, file=file, colors=colors)

  return df_norm


if __name__ == "__main__":
  
  df, calendar = load_cache()

  add_extra_time_units(df, inplace=True)
  print(df)

  # plot overall histograms
  # plot_weekday_dist(df)
  # plot_hourly_dist(df)

  # plot total submissions over time
  # dist = submissions_over_time(df)
  '''
  # plot hourly distribution for different parts of the quarter
  hist_comb = hist_hourly_quarter_edges(df, calendar, 3)
  plot_custom(hist_comb, 'line', 
                title='Hourly Submissions During Start vs End of Quarter', 
                file='hist_hourly_quarter_edges', xlabel='Hour', ylabel='Count',
                colors=['tab:green', 'tab:red'])

  # plot hourly distribution for different parts of the week
  hist_comb = hist_hourly_weekday_weeked(df)
  plot_custom(hist_comb, 'line', 
                title='Hourly Submissions Over the Weekdays', 
                file='hist_hourly_weekday_weeked', xlabel='Hour', ylabel='Count',
                colors=['tab:red', 'tab:green'])
  '''
  # sentiment analysis
  # df_sentiment_total = load_data(['sentiment_analysis.csv'])
  # quarter_spans = list(map(get_quarter_start_end, calendar))
  
  # stress words analysis

  df_collection = list()

  stress_words = load_corpus('stress_words_corpus.txt')
  df_result = keywords_over_quarters(stress_words, calendar, df, 
    title='Stress & Depression over typical Quarter', 
    file='plot_stress',
    colors=['lightgrey', 'tab:red', 'tab:green'])
  df_collection.append( ('Stress/Depression', df_result) )

  thirst_words = load_corpus('thirst_corpus.txt')
  df_result = keywords_over_quarters(thirst_words, calendar, df, 
    title='Thirstiness over typical Quarter', 
    file='plot_thirst',
    colors=['lightgrey', 'tab:purple'])
  df_collection.append( ('Thirstiness', df_result) )


  df_combined = pd.DataFrame() 
  for topic, df_result in df_collection:
      # try: df_combined.insert(0, 'Submissions', df_result['Submissions'])
      # except: pass
      df_combined.insert(len(df_combined.columns), topic, df_result['Keywords']-df_result['Submissions'])
  plot_custom(df_combined, 'bar', 
            extra_func=None,
            xlabel='Week of Quarter', ylabel='Activity Level',
            title='test',
            colors=['lightgrey', 'tab:red', 'tab:blue'])