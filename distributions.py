import pandas as pd
import pandas.api.types as ptypes
import matplotlib.pyplot as plt

days_of_week = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
mental_ill_keywords=['suicide','depression','fool','useless','dumb','cri','failure','fail']

def expand_timestamps(df):
  '''
  adds more columns for datetime attributes as categorials
  warning: inplace

  :param df: pandas DataFrame of df with 'timestamp' column of datetime format
  :return: same df with more columns
  '''
  assert 'timestamp' in df
  assert ptypes.is_datetime64_any_dtype(df['timestamp'])

  df['yr-week']=df.timestamp.dt.strftime('%y-%W')
  df['yr-month']=df.timestamp.dt.strftime('%y-%m')
  # df['year']=df.timestamp.dt.strftime('%y')
  df['weekday']=df.timestamp.dt.strftime('%a')
  df['weekday']=pd.Categorical(df['weekday'], categories=days_of_week, ordered=True)
  df['hour']=df.timestamp.dt.strftime('%H')

  return df

def plot_weekday_dist(df, keywords=None, title='Submissions per Day', file='dist_weekday.png', show=True):
  '''
  generates bar graph of confessions distribution over weekdays for entire dataframe

  :param df: pandas DataFrame of df with 'weekday' column
  :return: None
  '''
  assert 'weekday' in df
  # calculate
  weekday_dist = df.groupby('weekday')['content'].count().sort_index()
  # generate bat graph
  fig, ax = plt.subplots()
  ax.bar(weekday_dist.index, weekday_dist.values)
  # calculate y axis bottom
  scale_start = weekday_dist.values.min() - 0.15*(weekday_dist.values.max()-weekday_dist.values.min())
  ax.set_ylim(bottom=scale_start, auto=True)
  # set labels and generate figure
  ax.set_title(title)
  ax.set(xlabel='Week Day', ylabel='# of Submissions')
  fig.savefig(file)
  if show: plt.show()

if __name__ == "__main__":
  import glob
  from load_data import load_data
  print('Loading Excel spreadsheets...')
  df = load_data()
  expand_timestamps(df)
  print(df)
  plot_weekday_dist(df)