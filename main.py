import glob
import pickle
from load_data import load_data
from load_data import load_calendar
from distributions import add_extra_time_units
from distributions import histogram
from distributions import plot_histogram
from distributions import plot_custom
from distributions import plot_weekday_dist
from distributions import plot_hourly_dist
from distributions import submissions_over_time
from distributions import set_show_plots
from distributions import label_macro_size
import datetime as dt
import pandas as pd
import seaborn as sns

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

if __name__ == "__main__":
  
  df, calendar = load_cache()

  add_extra_time_units(df, inplace=True)
  print(df)

  # plot overall histograms
  # plot_weekday_dist(df)
  # plot_hourly_dist(df)

  # plot total submissions over time
  # dist = submissions_over_time(df)

  # plot hourly distribution for different parts of the quarter
  hist_comb = hist_hourly_quarter_edges(df, calendar, 2)
  plot_custom(hist_comb, 'line', 
                title='Hourly Submissions During Start vs End of Quarter', 
                file='hist_hourly_quarter_edges', xlabel='Hour', ylabel='Count')

  # plot hourly distribution for different parts of the week
  hist_comb = hist_hourly_weekday_weeked(df)
  plot_custom(hist_comb, 'bar', 
                title='Hourly Submissions Over the Weekdays', 
                file='hist_hourly_weekday_weeked', xlabel='Hour', ylabel='Count')

  

                