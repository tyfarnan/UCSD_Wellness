from load_data import load_data

if __name__ == "__main__":
  print('Loading Excel spreadsheets...')
  files = ["DatasetMain_use.xlsx", "DatasetOlder_use.xlsx"]
  df = load_data(files)
  print(df)