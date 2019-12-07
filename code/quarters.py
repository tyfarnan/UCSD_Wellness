import datetime as dt
import pandas as pd
import string
import re

from code.dist import add_extra_time_units
from code.dist import histogram
from code.dist import plot_histogram
from code.dist import plot_custom
from code.dist import plot_weekday_dist
from code.dist import plot_hourly_dist
from code.dist import submissions_over_time
from code.dist import label_macro_size
from code.dist import auto_reindex_lut

def get_quarter_start_end(quarter):
  '''
  returns the start and end dates of a quarter

  :param quarter: dict entry of a particular quarter
  :type quarter: dict
  :return: tuple(start, end)
  '''
  assert isinstance(quarter, dict)
  try: mask_begin = quarter['start'] # try official start
  except: mask_begin = quarter['begins'] # else use instruction start
  mask_end = quarter['end'] # get end of quarter (after finales)
  return (mask_begin, mask_end)

def hist_hourly_quarter_edges(df, calendar, span=3):
  '''
  compares entries spanning the beginning vs end of the average quarter
  by hour of the day

  :param df: confessions with timestamp index
  :type df: pandas DataFrame
  :param calendar: list of dict entries of quarters
  :type calendar: list
  :param span: number of weeks to span on either end
  :type span: int
  :return: pandas Series
  '''
  assert isinstance(df, pd.DataFrame)
  assert isinstance(calendar, list)
  assert isinstance(span, int)

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

def hist_hourly_weekday_weekend(df):
  '''
  compares entries on weekdays vs weekends
  by hour of the day
  note: friday is defined as a weekend day

  :param df: confessions with timestamp index
  :type df: pandas DataFrame
  :return: pandas Series
  '''
  assert isinstance(df, pd.DataFrame)

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
  '''
  counts how many times words from a given corpus can be found in a submission

  :param words: corpus of words to use
  :type words: list of str
  :param content: string format of confessions content
  :type content: str
  :return: int count
  '''
  assert isinstance(words, list)
  assert isinstance(content, str)

  filtered = list(map(lambda x: x.strip().translate(str.maketrans('', '', string.punctuation)), content.split()))
  return sum(map(lambda word: word in words, filtered))

def keywords_over_quarters(corpus, calendar, df, title=None, file=None, colors=['lightgrey', 'tab:red', 'tab:green'], show=True):
  '''
  finds and plots the instances of corpus words over an average quarter

  :param corpus: corpus of words to use
  :type corpus: list of str
  :param calendar: list of dict entries of quarters
  :type calendar: list
  :param df: confessions with timestamp index
  :type df: pandas DataFrame
  :param title: title of plot (optional)
  :type title: str
  :param file: filename to save plot (optional)
  :type file: str
  :param colors: set of colors to use for plot (optional)
  :type colors: list of str
  :param show: show the plot interactively (optional)
  :type show: bool
  :return: normalized histogram in DataFrame use to plot the graph
  '''
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
                title=title, file=file, colors=colors,
                show=show)

  return df_norm


if __name__ == "__main__":

  # DEMO code of this module.

  import glob
  from load_data import load_data
  from load_data import load_calendar
  from load_data import load_corpus
  from load_data import load_cache
  
  df, calendar = load_cache()
  add_extra_time_units(df, inplace=True)
  print(df)

  # plot overall histograms
  plot_weekday_dist(df, show=False)
  plot_hourly_dist(df, show=False)

  # plot total submissions over time
  dist = submissions_over_time(df, show=False)

  # plot hourly distribution for different parts of the quarter
  hist_comb = hist_hourly_quarter_edges(df, calendar, span=3)
  plot_custom(hist_comb, 'line', 
                title='Hourly Submissions During Start vs End of Quarter', 
                file='hist_hourly_quarter_edges', xlabel='Hour', ylabel='Count',
                colors=['tab:green', 'tab:red'],
                show=False)

  # plot hourly distribution for different parts of the week
  hist_comb = hist_hourly_weekday_weekend(df)
  plot_custom(hist_comb, 'line', 
                title='Hourly Submissions Over the Weekdays', 
                file='hist_hourly_weekday_weekend', xlabel='Hour', ylabel='Count',
                colors=['tab:red', 'tab:green'],
                show=False)


  # word corpus analysis
  df_collection = list()

  # individual corpus analysis plot
  stress_words = load_corpus('data/stress_corpus.txt')
  df_result = keywords_over_quarters(stress_words, calendar, df, 
    title='Stress & Depression over typical Quarter', 
    file='plot_stress',
    colors=['lightgrey', 'tab:red', 'tab:green'],
    show=False)
  df_collection.append( ('Stress/Depression', df_result) )

  thirst_words = load_corpus('data/thirst_corpus.txt')
  df_result = keywords_over_quarters(thirst_words, calendar, df, 
    title='Romantic Attraction over typical Quarter', 
    file='plot_thirst',
    colors=['lightgrey', 'tab:purple'],
    show=False)
  df_collection.append( ('Romantic Attraction', df_result) )

  # combined corpus analysis plot
  df_combined = pd.DataFrame() 
  for topic, df_result in df_collection:
      df_combined.insert(len(df_combined.columns), topic, df_result['Keywords']-df_result['Submissions'])
  plot_custom(df_combined, 'bar', 
            extra_func=None,
            xlabel='Week of Quarter', ylabel='Activity Level',
            title='test',
            colors=['lightgrey', 'tab:red', 'tab:blue'],
            show=False)