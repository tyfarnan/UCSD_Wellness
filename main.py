import glob
from load_data import *

if __name__ == "__main__":
  
  print('Loading Excel spreadsheets...')
  # files = ["DatasetMain_use.xlsx", "DatasetOlder_use.xlsx"]
  files = glob.glob('./*.xlsx')
  df = load_data(files)
  print(df)