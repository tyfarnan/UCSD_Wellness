# UCSD Confessions Analysis

*Author: Curtis Lee, Meihan Zhang, Tyler Farnan*
**Summary:** Hello! This repository contains a jupyter notebook demonstration of exploratory sentiment analysis on UCSD Confessions, an anomyous online forum for students.


## Dataset

[UCSD Confessions Public Archive](https://drive.google.com/drive/folders/1vTfogzYi7MC1p0tpkEeBMRFgqNnYm8Wl?usp=sharing)

## Third Party Modules Used

Numpy, Pandas, Matplotlib, Seaborn, Glob, NLTK, TextBlob, Scikit-learn, Gensim, Wordcloud

## Requirements for Sentiment Analysis:

To run the sentiment analysis portion, extra features are needed:
```
$ python -m textblob.download_corpora
```

Then run python and download the data set:
```
>>> import nltk
>>> nltk.download('stopwords')
```

## File Stucture

This repo is broken down into multiple folders.
* **code** contains all the necessary python modules
  * some of the code modules may use each other
* **data** contains all the dataset files used in this project
  * xlsx files contain the UCSD confessions data
  * academic_calendar.json contains breakdown of the UCSD Quarter System
  * _corpus.txt files contains list of words used for keyword trend analysis
  * .pkl and .dat cache files generated one time for fast access later
  * other files are generated by the lexicon analysis process
* **output** is a dedicated folder for the graphs generated by the process

*Note: some of the code modules have demo features and can be run independantly, but must in the root directory of repo*

for example:
` $ python load_data.py `

## How to run code:

1. Clone the repo
2. Install all the necessary third party modules and dependancies
3. **Please see main.ipynb for full usage details**