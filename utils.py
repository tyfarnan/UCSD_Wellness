import xlrd 
import sqlite3
import datetime
import itertools
from collections import defaultdict
from dataclasses import dataclass
import matplotlib
import matplotlib.dates
import matplotlib.pyplot

def create_table():
  conn = sqlite3.connect('AA_db.sqlite')
  cur = conn.cursor()
  cur.execute('''CREATE TABLE confessions (
    id int NOT NULL AUTO_INCREMENT, 
    timestamp VARCHAR,
    status VARCHAR,
    content VARCHAR,
    timestamp VARCHAR,
    tags VARCHAR,
    comment VARCHAR,
    note VARCHAR,
    contact VARCHAR,
    contact VARCHAR,
    )''')
  conn.commit()
  conn.close()


def parse_timestamp(timestamp):
  '''
  convert confessions format timestamp string to datetime object
  else if datetime object passed, simply returns itself
  otherwise raises TypeError
  
  :param timestamp_str: confessions spreadsheet timestamp string
  :type timestamp_str: str
  :return: returns datetime object
  '''
  assert(timestamp != None)
  if isinstance(timestamp, datetime.datetime):
    return timestamp
  elif isinstance(timestamp, tuple):
    return datetime.datetime(*timestamp[0:6])
  elif isinstance(timestamp, str):
    try:
      return datetime.datetime.strptime(timestamp, '%m/%d/%Y %H:%M:%S')
    except:
      return datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
  else:
    raise TypeError("timestamp "+str(timestamp)+" is neither datetime type nor timestamp-as-string")

def get_rows_from_sheets(filenames, autotimestamp=True, timestamp_name='timestamp'):
  '''
  creates a generator for all the rows from every sheet of every workbook
  uses the header row (1st row) of each sheet for the key values
  
  :param filenames: filenames of all the workbooks
  :type filenames: list of str
  :return: a generator for the rows
  '''
  assert(isinstance(filenames, list)) # ensure list
  for file in filenames:
    assert(isinstance(file, str)) # ensure list has strs
    wb = xlrd.open_workbook(file) 
    sheets_list = wb.sheets()
    for sheet in sheets_list:
      rows = sheet.get_rows() # returns row generator per sheet
      timestamp_index = None # store timestamp index for autotimestamp option

      # deconstruct header row to use as dict keys
      header = next(rows) # use header row
      fields = list() # name of fields
      for index, field in enumerate(header):
        assert(isinstance(field.value, str)) # ensure field name representable as str
        if field.value == '': # skip empty entries
          continue
        if field.value == timestamp_name:
          timestamp_index = index
        fields.append(field.value) # append str value of field
      
      # generator for dict
      while True:
        try:
          row = next(rows)
          values = list()
          for index, cell in enumerate(row):
            # skip empty entries
            if cell.ctype==0 or cell.value == '':
              continue
            # decode excel date format first
            if cell.ctype==3:
              value = xlrd.xldate_as_tuple(cell.value, wb.datemode)
            else:
              value = cell.value
              if autotimestamp and index==timestamp_index:
                value = parse_timestamp(value)
            # append cell value
            values.append(value)
          
          # zip into dict
          row_as_dict = dict(zip(fields, values))
          yield row_as_dict

        except StopIteration:
          break

@dataclass
class WeeklyCountOccurance:
  def __init__(self, week_num: int, timestamp: datetime.datetime, count: int):
    self.week_num = week_num
    self.timestamp = timestamp
    self.count = count

@dataclass
class WordCountOccurance:
  def __init__(self, timestamp: datetime.datetime, count: int):
    self.timestamp = timestamp
    self.count = count

# def submission_count_weekly(row_gen, start_datetime, weeks, once_per_post=False):
#   '''
#   '''
#   week_begins = [ start_datetime+datetime.timedelta(weeks=week) for week in range(weeks) ]
#   for row in row_gen:
    


#   weekly_dict = defaultdict(int)

#   for word in wordcount_dict:
#     occurance_list = wordcount_dict[word]
#     occurange_list_weekly = list()
#     for week in range(weeks):
#       weekly_occurance = 0
#       week_begin = start_datetime + datetime.timedelta(weeks=week)
#       week_end = start_datetime + datetime.timedelta(weeks=week+1)
#       for occurance in occurance_list:
#         timestamp = occurance.timestamp
#         if timestamp >= week_begin and timestamp <= week_end:
#           if once_per_post:
#             weekly_occurance += 1
#           else:
#             weekly_occurance += occurance.count
#       occurange_list_weekly.append( WordCountOccurance(week_begin, weekly_occurance) ) 
#     weekly_dict[word] = occurange_list_weekly
#   return weekly_dict

def wordcount(row_gen):
  '''
  generates a dictonary with words as keys and list of WordCountOccurance objects as values
  each WordCountOccurance corresponds to a single post and contains a timestamp and number of occurances
  
  :param row_gen: rows corresponding to confessions as dict objects
  :type row_gen: list or generator of rows
  :return: dictionary of wordcounts
  '''
  wordcounts_global = defaultdict(list)
  for row in row_gen:
    content = row['content']
    timestamp = parse_timestamp(row['timestamp'])
    
    wordcounts_local = defaultdict(int)
    words = content.split()
    for word in words:
      wordcounts_local[word] += 1

    for word in wordcounts_local:
      wordcount = wordcounts_local[word]
      wordcount_obj = WordCountOccurance(timestamp, wordcount)
      wordcounts_global[word].append(wordcount_obj)

  return wordcounts_global

def wordcount_totals(wordcount_dict, once_per_post=False):
  '''
  absolute total historgrams of word occurances
  '''
  histogram = defaultdict(int)
  for word in wordcount_dict:
    occurance_list = wordcount_dict[word]
    # each item occurance list corresponds to single post where the word appears
    if once_per_post:
      # simply count the number of posts it appears
      histogram[word] = len(occurance_list)
    else:
      # accumulate total occurances in all posts it appears
      for occurance in occurance_list:
         histogram[word] += occurance.count
  return histogram
  
def wordcount_weekly(wordcount_dict, start_datetime, weeks, once_per_post=False):
  '''
  '''
  weekly_dict = defaultdict(int)
  for word in wordcount_dict:
    occurance_list = wordcount_dict[word]
    occurange_list_weekly = list()
    for week in range(weeks):
      weekly_occurance = 0
      week_begin = start_datetime + datetime.timedelta(weeks=week)
      week_end = start_datetime + datetime.timedelta(weeks=week+1)
      for occurance in occurance_list:
        timestamp = occurance.timestamp
        if timestamp >= week_begin and timestamp <= week_end:
          if once_per_post:
            weekly_occurance += 1
          else:
            weekly_occurance += occurance.count
      occurange_list_weekly.append( WordCountOccurance(week_begin, weekly_occurance) ) 
    weekly_dict[word] = occurange_list_weekly
  return weekly_dict



def plot_wordcount(wordcount_dict, words, merge_words=True):
  '''
  plots histograms  
  '''
  if merge_words:
    date_dict = defaultdict(int)
    for word in words:
      occurance_list = wordcount_dict[word]
      for occurance in occurance_list:
        date_dict[occurance.timestamp] += occurance.count
    counter=0
    for date in date_dict:
      print(date, 'week', counter, ':', date_dict[date])
      counter += 1
    dates, values = list(date_dict.keys()), list(date_dict.values())
  else:
    for word in words:
      occurance_list = wordcount_dict[word]
      dates = [ x.timestamp for x in occurance_list ]
      dates = matplotlib.dates.date2num (dates)
      values = [ x.count for x in occurance_list ]  
  matplotlib.pyplot.plot_date(dates, values)
  matplotlib.pyplot.title(str(words))
  matplotlib.pyplot.show()


def fix_timestamps(rows, fieldname='timestamp'):
  '''
  runs parse_timestamp on timestamp field of rows
  '''
  for row in rows:
    row[fieldname] = parse_timestamp(row[fieldname])
  return rows

def sort_rows_by_time(rows):
  '''
  sorts rows by timestamp
  '''
  rows.sort(key=lambda row: row['timestamp'])
  return rows

def count_total_entrys(rows, datetime_begin, datetime_end):
  '''
  returns number of entries found in time interval
  '''


if __name__ == "__main__":
  # convert()
  timestamp_str = '10/29/2018 2:20:35'
  # timestamp_str = '2018-10-31 19:08:23'
  timestamp = parse_timestamp(timestamp_str)
  print(timestamp)

  print('Loading Excel spreadsheets...')
  files = ["DatasetMain_use.xlsx", "DatasetOlder_use.xlsx"]
  row_gen = get_rows_from_sheets(files)
  # rows = list(row_gen)
  # fix_timestamps(rows)
  # sort_rows_by_time(rows)
  # row_gen = get_rows_from_sheets(files)

  print('Obtaining global word counts...')
  wordcountdict = wordcount(row_gen)

  print('Analyzing Winter 2019 word counts...')
  winter_weekly = winter_weekly = wordcount_weekly(wordcountdict, datetime.datetime(2018, 12, 31), 12)

  plot_wordcount(winter_weekly, ['stress', 'failure', 'fail', 'die', 'depressed', 'depression', 'sad', 'cry', 'crying', 'dumb'])