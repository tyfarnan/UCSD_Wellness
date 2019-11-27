import collections
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

def load_data(files):
  '''
  loads confessions data from all sheets in excel files and convert to
  pandas dataframe indexed by timestamp

  :param files: name/path to the excel files to load
  :type files: list of str
  '''
  # iterate all files and get sheets
  sheets = list()
  for file in files:
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
  # set the timestamp as index and sort
  df.set_index('timestamp', inplace=True)
  df.sort_index(inplace=True)

  return df


if __name__ == "__main__":
  print('Loading Excel spreadsheets...')
  files = ["DatasetMain_use.xlsx", "DatasetOlder_use.xlsx"]
  df = load_data(files)
  print(df)