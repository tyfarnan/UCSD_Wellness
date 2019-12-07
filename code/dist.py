import pandas as pd
import pandas.api.types as ptypes
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
import datetime as dt

strptime=dt.datetime.strptime
strftime=dt.datetime.strftime

show_plots = True

# plot tweaks
label_macro_size = 'x-large'
weekdays = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
auto_reindex_lut = {
  'weekday': lambda x: weekdays[x],
  'hour': lambda x: strptime(str(x), "%H").strftime("%I %p"),
  'month': lambda x: strptime(str(x), "%m").strftime("%b")
}

def add_extra_time_units(df, inplace=False):
  '''
  adds useful categories not found in standard datetime

  :param df: pandas DataFrame of confessions with timestamp index
  :param inplace: modify existing dataframe if True
  :return: None
  '''
  assert ptypes.is_datetime64_any_dtype(df.index)
  if inplace: df_new = df
  else: df_new = df.copy()
  df_new['yr-week']=df.index.strftime('%y-%W')
  df_new['yr-month']=df.index.strftime('%y-%m')
  return df

def histogram(df, category, reindexer='auto'):
  '''
  generates histogram based on specified datetime category
  useful for making frequency plots

  :param df: confessions with timestamp index
  :type df: pandas DataFrame
  :param category: the datetime attribute to group by
  :type category: str name of attribute
  :param reindexer: function to reindex x labels (optional)
  :type reindexer: function
  :return: histogram in pandas Series
  '''
  if reindexer=='auto':
    if category in auto_reindex_lut:
      reindexer = auto_reindex_lut[category]
    else:
      reindexer = None
  try:
    category = df[category]
  except:
    # ensure DataFrame has datetime index, and proper datetime category passed
    assert ptypes.is_datetime64_any_dtype(df.index)
    assert hasattr(df.index, category)
    # group by category and generate histogram
    category = getattr(df.index, category)
  dist = df.groupby(category).count().content
  if reindexer: dist.index = [reindexer(i) for i in dist.index]
  return dist

def plot_histogram_render_manual(ax, dist):
  '''
  manual plotting of bar chart

  :param ax: axes to plot on
  :type ax: pyplot axes
  :param dist: pandas histogram 
  :type dist: pandas Series
  '''
  assert ax != None
  ax.bar(dist.index, dist.values)
  # calculate y axis bottom
  scale_start = dist.values.min() - 0.15*(dist.values.max()-dist.values.min())
  ax.set_ylim(bottom=scale_start, auto=True)
  if len(dist.index) > 12:
    ax.set_xticklabels(dist.index, rotation=45)
    # https://stackoverflow.com/questions/20337664/cleanest-way-to-hide-every-nth-tick-label-in-matplotlib-colorbar
    for label in ax.xaxis.get_ticklabels()[::len(dist.index)//12]: label.set_visible(False)

def plot_histogram(df, category, reindexer='auto', title=None, file=None, xlabel=None, ylabel=None, show=show_plots):
  '''
  wrapper function for histogram() to plot the histogram

  :param df: of confessions with timestamp index
  :type df: pandas DataFrame
  :param category: the datetime attribute to group by
  :type category: str name of attribute
  :param reindexer: function to reindex x labels (optional)
  :type reindexer: function
  :param title: title of plot (optional)
  :type title: str
  :param file: filename to save plot (optional)
  :type file: str
  :param xlabel: str label for x axis (optional)
  :type xlabel: str
  :param ylabel: str label for y axis (optional)
  :type ylabel: str
  :param show: show the plot interactively (optional)
  :type show: bool
  :return: histogram in pandas Series
  '''
  assert isinstance(df, pd.DataFrame)
  fig,ax=plt.subplots()

  dist = histogram(df, category, reindexer)

  plot_histogram_render_manual(ax, dist)

  # generate default labels
  if not title: title = 'Total Count by '+category.capitalize()
  if not xlabel: xlabel = category.capitalize()
  if not ylabel: ylabel = 'Count'
  # set labels
  if title: ax.set_title(title, size=label_macro_size)
  if xlabel: ax.set_xlabel(xlabel, size=label_macro_size) 
  if ylabel: ax.set_ylabel(ylabel, size=label_macro_size)
  # generate figure
  fig.tight_layout()
  if file: fig.savefig(file)
  if show: plt.show()

  return dist


def plot_weekday_dist(df, keywords=None, title='Submissions per Weekday', file='dist_weekday.png', show=show_plots):
  '''
  generates bar graph of confessions distribution over weekdays for entire dataframe

  :param df: pandas DataFrame of confessions with timestamp index
  :type df: pandas DataFrame
  :return: None
  '''
  assert isinstance(df, pd.DataFrame)
  return plot_histogram(df, 'weekday', title=title, file=file, show=show)

def plot_hourly_dist(df, keywords=None, title='Submissions per Hour of Day', file='dist_hourly.png', show=show_plots):
  '''
  generates bar graph of confessions distribution over hours of the day for entire dataframe

  :param df: pandas DataFrame of confessions with timestamp index
  :type df: pandas DataFrame
  :return: None
  '''
  assert isinstance(df, pd.DataFrame)
  return plot_histogram(df, 'hour', title=title, file=file, show=show)

def plot_custom(df, plot_type, extra_func=None, colors=None, title=None, file=None, xlabel=None, ylabel=None, show=show_plots):
  '''
  
  wrapper function to plot more generic things from pandas dataframe

  :param df: of confessions with timestamp index
  :type df: pandas DataFrame
  :param category: the datetime attribute to group by
  :type category: str name of attribute
  :param reindexer: function to reindex x labels (optional)
  :type reindexer: function
  :param title: title of plot (optional)
  :type title: str
  :param file: filename to save plot (optional)
  :type file: str
  :param xlabel: str label for x axis (optional)
  :type xlabel: str
  :param ylabel: str label for y axis (optional)
  :type ylabel: str
  :param show: show the plot interactively (optional)
  :type show: bool
  :return: histogram in pandas Series
  '''  
  assert isinstance(df, pd.DataFrame)
  assert hasattr(df.plot, plot_type)
  plot_func = getattr(df.plot, plot_type)

  # group by category and generate histogram
  ax = plot_func(color=colors)
  fig = ax.get_figure()

  if len(df.index) > 12 and plot_type=='bar':
    ax.set_xticklabels(df.index, rotation=45)
    # https://stackoverflow.com/questions/20337664/cleanest-way-to-hide-every-nth-tick-label-in-matplotlib-colorbar
    for label in ax.xaxis.get_ticklabels()[::len(df.index)//12]: label.set_visible(False)

  if extra_func: ax = extra_func(ax)

  # generate default labels
  if not title: title = 'Total Count'
  if not xlabel: xlabel = 'Time'
  if not ylabel: ylabel = 'Count'
  # set labels
  if title: ax.set_title(title, size=label_macro_size)
  if xlabel: ax.set_xlabel(xlabel, size=label_macro_size) 
  if ylabel: ax.set_ylabel(ylabel, size=label_macro_size)
  ax.legend(fontsize='large')
  # generate figure
  fig.tight_layout()
  if file: fig.savefig(file)
  if show: plt.show()

def submissions_over_time(df_orignal, dt_range=None, unit='date', title='Submissions over Time', file='plot_time.png', show=show_plots):
  '''
  generates plot of confessions per week for entire dataset

  :param df_orignal: pandas DataFrame of confessions with timestamp index
  :return: pandas Series of the distribution 
  '''
  assert ptypes.is_datetime64_any_dtype(df_orignal.index)

  df = df_orignal.copy()
  if dt_range:
    # https://stackoverflow.com/questions/29370057/select-dataframe-rows-between-two-dates
    mask = (df.index > dt_range[0]) & (df.index <= dt_range[1])
    df = df.loc[mask]

  fig,ax=plt.subplots()
  # calculate
  # dstart = df.index[0]
  label_format = '%b %d\n%Y'
  dist = pd.DataFrame( df.groupby(df.index.date).count().content )
  dist.columns=['total']
  dist.index = [dt.datetime(x.year, x .month, x.day) for x in dist.index] # convert date to datetime
  dist.index.name = 'date'
  dist['index_str'] = dist.index.strftime(label_format)

  # generate plot
  ax.plot(dist.index_str, dist.total)

  # set x labels
  ax.xaxis.set_major_locator(LinearLocator(numticks=10))
  ax.autoscale(enable=True, axis='x', tight=True)

  # set labels
  ax.set_title(title, size=label_macro_size)
  ax.set_xlabel('Day', size=label_macro_size) 
  ax.set_ylabel('Submissions per Day', size=label_macro_size)
  ax.grid()

  # generate figure
  fig.tight_layout()
  fig.savefig(file)
  if show: plt.show()

  return dist

if __name__ == "__main__":
  import glob
  from load_data import load_data
  from load_data import load_calendar
  print('Loading Excel spreadsheets...')
  df = load_data()
  add_extra_time_units(df, inplace=True)
  print(df)

  # plot histograms
  dist = plot_weekday_dist(df, file=None)
  print(dist)
  dist = plot_hourly_dist(df, file=None)
  print(dist)

  # plot total submissions over time
  dist = submissions_over_time(df, file=None)
  print(dist)
    
