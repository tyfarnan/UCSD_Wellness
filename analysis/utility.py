#Import Modules
import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
import re
import os

import csv
import pandas as pd
import numpy as np

from gensim.models.word2vec import LineSentence
from nltk.corpus import subjectivity
nltk.download('vader_lexicon')
# first, we import the relevant modules from the NLTK library
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# !pip install wordcloud

import nltk.data;
from gensim.models import word2vec;
from sklearn.cluster import KMeans;
from sklearn.neighbors import KDTree;
import pandas as pd;
import numpy as np;
import os;
import re;
import logging;
import sqlite3;
import time;
import sys;
import multiprocessing;
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt;
from itertools import cycle;

from nltk.corpus.reader import WordListCorpusReader
from nltk.corpus.reader.api import *
nltk.download('opinion_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import opinion_lexicon
from nltk.corpus import stopwords

from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict

#convert txt to csv files
def txt_to_csv(txt_file, new_csv_file, header=int):   
    """""
    :param txt_file: current txt file
    :type txt_file: str
    :param new_csv_file: destination file
    :type new_csv_file: str
    :returns: new csv file
    """
    assert isinstance(txt_file, str)
    assert isinstance(new_csv_file, str)

    in_txt = csv.reader(open(txt_file, "r"), delimiter = '\t')
    out_csv = csv.writer(open(new_csv_file, 'w'))

    out_csv.writerows(in_txt)
    
    lex_csv = pd.read_csv(str(new_csv_file),header=header)
    
    return lex_csv

def clean_text(all_comments, out_name):
    """""

    :param all_comments: dirty csv
    :type all_comments: str
    :param new_csv_file: clean 
    :type new_csv_file: str
    :returns: new csv file
    """
    
    assert isinstance(out_name, str)
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle');
    
    stop_words = list(set(stopwords.words('english')))
    
    out_file = open(out_name, 'w');
    
    for pos in range(len(all_comments)):
        
        #Get the comment
        val = all_comments.iloc[pos];
        
        #Normalize tabs and remove newlines
        no_tabs = str(val).replace('\t', ' ').replace('\n', '');
        
        #Remove all characters except A-Z and a dot.
        alphas_only = re.sub("[^a-zA-Z\.]", " ", no_tabs);
        
        #Remove punctuation
        no_punc = re.sub(r'[^(a-zA-Z)\s]','', alphas_only)
        
        #Normalize spaces to 1
        multi_spaces = re.sub(" +", " ", no_punc);
        
        #Strip trailing and leading spaces
        no_spaces = multi_spaces.strip();
        
        #Normalize all charachters to lowercase
        clean_text = no_spaces.lower();
        
        #Get sentences from the tokenizer, remove the dot in each.
        sentences = tokenizer.tokenize(clean_text);
        
        stopped = [w for w in sentences if not w in stop_words]
        
        sentences = [re.sub("[\.]", "", sentence) for sentence in sentences];
        
            #If the text has more than one space (removing single word comments) and one character, write it to the file.
        out_file.write("%s\n" % sentences)
                

    out_file.close();
    return None
    
def compute_pos_neg_scores(clean_content): 
    """""

    :param clean_content: pre-processed confessions
    :type all_comments: list
    :returns: None
    """
    assert isinstance(clean_content, list)

    #nltk opinion lexicon
    pos_lex = opinion_lexicon.positive()
    neg_lex = opinion_lexicon.negative()
    pn_lex_score = defaultdict(int)
    
    for i,note in tqdm(enumerate(clean_content)):
        pn_lex_score[i] = np.array([0.,0.])
        note = re.sub("[^\w]", " ",  note).split()
        for word in note:
            if word in pos_lex:
                pn_lex_score[i] += np.array([1.,0.])
            elif word in neg_lex:
                pn_lex_score[i] += np.array([0.,1.])
                
    output = open('pn_lex_score.pkl', 'wb')
    pickle.dump(pn_lex_score, output)
    output.close()
    return None

def compute_VAD_scores(clean_content, vad_lex):
    """""

    :param clean_content: pre-processed confessions
    :type all_comments: list
    :returns: None
    """
    assert isinstance(clean_content, list)
    
    #NRC VAD lexicon
    all_VAD_scores = defaultdict(int)
    # clean_content = clean_comments[0].tolist()
    for i,note in tqdm(enumerate(clean_content)):
        all_VAD_scores[i] = np.array([0.,0.,0.])
        note = re.sub("[^\w]", " ",  note).split()
        for word in note:
            if word in list(vad_lex['Word']):
                all_VAD_scores[i] += np.array([vad_lex['Valence'][i],vad_lex['Arousal'][i],vad_lex['Dominance'][i]])
    
    output = open('all_VAD_scores.pkl', 'wb')
    pickle.dump(all_VAD_scores, output)
    output.close()
    return None

def compute_nltk_vader_scores(clean_content):
    """""

    :param clean_content: pre-processed confessions
    :type all_comments: list
    :returns: None
    """
    assert isinstance(clean_content, list)
    
    # next, we initialize VADER so we can use it within our Python script
    sid = SentimentIntensityAnalyzer()

    all_nltk_vader_scores = defaultdict(int)
    # clean_content = clean_comments[0].tolist()
    for i,note in tqdm(enumerate(clean_content)):
        scores = sid.polarity_scores(note)
        all_nltk_vader_scores[i] = np.array([scores['neg'],scores['neu'],scores['pos'],scores['compound']])

    output = open('all_nltk_vader_scores.pkl', 'wb')
    pickle.dump(all_nltk_vader_scores, output)
    output.close()
    return None

def plot_vad_weekday_scores(new): 
    """""

    :param clean_content: pre-processed confessions
    :type all_comments: list
    :returns: None
    """
    assert isinstance(new,pd.DataFrame)
    # calculate total scores and counts
    weekday_valence_sum=new.groupby('weekday')['Valence'].sum()
    weekday_valence_count=new.groupby('weekday')['Valence'].count()
    
    weekday_arousal_sum=new.groupby('weekday')['Arousal'].sum()
    weekday_arousal_count=new.groupby('weekday')['Arousal'].count()
    
    weekday_dominance_sum=new.groupby('weekday')['Dominance'].sum()
    weekday_dominance_count=new.groupby('weekday')['Dominance'].count()
    
    valence= weekday_valence_sum/weekday_valence_count
    arousal = weekday_arousal_sum/weekday_arousal_count
    dominance = weekday_dominance_sum/weekday_dominance_count
    
    #specify plot
    fig, ax = plt.subplots()
    index = np.arange(7)
    bar_width =0.5
    w = 0.35
    ax.bar(index-w, valence.values-10, width=w, color='b', align='center',label="Valence")
    ax.bar(index, arousal.values-10, width=w, color='g', align='center', label="Arousal")
    ax.bar(index+w, dominance.values-10, width=w, color='r', align='center', label="Dominance")
    # ax.bar(list(weekday_distri.index))
    # plt.bar(index, weekday_valence.values)
    # plt.bar(index+bar_width, weekday_arousal.values)
    # plt.bar(index+bar_width*2, weekday_dominance.values)
    # ax.set(xlabel='weekday', ylabel='VAD score')
    ax.grid(False)
    plt.ylabel("VAD Scores (normalized)")
    plt.xlabel("Day of the Week")
    plt.xticks(index,list(weekday_valence_sum.index))
    plt.legend()
    plt.title('Valence, Arousal, and Dominance over typical Week ')
    fig.savefig("weekday.png")
    plt.show()
    return None

def plot_nltk_vader_scores(df):
    """""

    :param clean_content: pre-processed confessions
    :type all_comments: list
    :returns: None
    """
    assert isinstance(df,pd.DataFrame)
    
    #setup plots
    hour_neg_sum=df.groupby('hour')['neg'].sum()
    hour_neg_count=df.groupby('hour')['neg'].count()
    neg = hour_neg_sum/hour_neg_count

    hour_pos_sum=df.groupby('hour')['pos'].sum()
    hour_pos_count=df.groupby('hour')['pos'].count()
    pos = hour_pos_sum/hour_pos_count

    hour_neu_sum=df.groupby('hour')['neu'].sum()
    hour_neu_count=df.groupby('hour')['neu'].count()
    neu = hour_neu_sum/hour_neu_count

    hour_compound_sum=df.groupby('hour')['compound'].sum()
    hour_compound_count=df.groupby('hour')['compound'].count()
    compound = hour_compound_sum/hour_compound_count
    fig, ax = plt.subplots()
    index = np.arange(24)
    bar_width =0.5
    w = 0.34
    ax.bar(index+w, neg.values, width=w, color='r', align='center',label='negative')
    # ax.bar(index-(w/2), neu.values, width=w, color='y', align='center')
    ax.bar(index, compound.values, width=w, color='g', align='center',label='compound')
    ax.bar(index-w, pos.values, width=w, color='b', align='center',label='positive')

    # ax.bar(list(weekday_distri.index))
    ax.grid(False)
    # ax.set_xticklabels(np.arange(24))
    plt.ylabel("VAD Polarity Scores (normalized)")
    plt.xlabel("Hour of the day")
    plt.xticks(np.arange(0,25, step=4))

    # ax.grid(False)
    # plt.xticks(index,list(compound.index))

    plt.title('Polarity of Confessions Over 24 Hours')
    plt.legend(loc=1, prop={'size':12 })
    # plt.xticks(np.arange(12), calendar.month_name[1:13], rotation=20)
    fig.savefig("weekday_nltk_vader.png")
    plt.show()
    return None

def train_word2vec(csv_filename_string, num_f, min_wc, ctxt,ds):   
    """""

    :param csv_filename_string: file
    :type csv_filename_string: str
    :param num_f: features
    :type num_f: int
    :param min_wc: min word count
    :type min_wc: int
    :param ctxt: context
    :type ctxt: int
    :param ds: downsampling
    :type ds: float
    :returns: None
    """
    assert isinstance(csv_filename_string,str)
    assert isinstance(num_f, int)
    assert isinstance(min_wc, int)
    assert isinstance(ctxt, int)
    assert isinstance(ds, float)
    
    
    start = time.time();
    #Set the logging format to get some basic updates.
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)
    # Set values for various parameters
    num_features = num_f;    # Dimensionality of the hidden layer representation
    min_word_count = min_wc;   # Minimum word count to keep a word in the vocabulary
    num_workers = multiprocessing.cpu_count();       # Number of threads to run in parallel set to total number of cpus.
    context = ctxt       # Context window size (on each side)                                                       
    downsampling = ds   # Downsample setting for frequent words
    # Initialize and train the model. 
    #The LineSentence object allows us to pass in a file name directly as input to Word2Vec,
    #instead of having to read it into memory first.
    print("Training model...");
    model = word2vec.Word2Vec(LineSentence(csv_filename_string), workers=num_workers, \
                size=num_features, min_count = min_word_count, \
                window = context, sample = downsampling);
    # We don't plan on training the model any further, so calling 
    # init_sims will make the model more memory efficient by normalizing the vectors in-place.
    model.init_sims(replace=True);
    # Save the model
    model_name = "model_"+str(csv_filename_string);
    model.save(model_name);
    print('Total time: ' + str((time.time() - start)) + ' secs')
    return model

def clustering_on_wordvecs(word_vectors, num_clusters):
    """""

    :param clean_content: pre-processed confessions
    :type all_comments: list
    :returns: None
    """
    assert isinstance(word_vectors, np.ndarray)
    assert isinstance(num_clusters, int)
    
    # Initalize a k-means object and use it to extract centroids
    kmeans_clustering = KMeans(n_clusters = num_clusters, init='k-means++');
    idx = kmeans_clustering.fit_predict(word_vectors);
    
    return kmeans_clustering.cluster_centers_, idx;

def get_top_words(index2word, k, centers, wordvecs):
    """""

    :param clean_content: pre-processed confessions
    :type all_comments: list
    :returns: None
    """
    assert isinstance(index2word, list)
    assert isinstance(k, int)
    assert isinstance(centers, np.ndarray)
    assert isinstance(wordvecs, np.ndarray) 
    
    tree = KDTree(wordvecs);
    #Closest points for each Cluster center is used to query the closest 20 points to it.
    closest_points = [tree.query(np.reshape(x, (1, -1)), k=k) for x in centers];
    closest_words_idxs = [x[1] for x in closest_points];
    #Word Index is queried for each position in the above array, and added to a Dictionary.
    closest_words = {};
    for i in range(0, len(closest_words_idxs)):
        closest_words['Cluster #' + str(i)] = [index2word[j] for j in closest_words_idxs[i][0]]
        
    #A DataFrame is generated from the dictionary.
    print(closest_words['Cluster #0'])
    df = pd.DataFrame(closest_words);
    df.index = df.index+1
    return df;

def display_cloud(cluster_num, cmap, top_words, label=None):
    """""

    :param clean_content: pre-processed confessions
    :type all_comments: list
    :returns: None
    """
    assert isinstance(cluster_num, int)
    assert isinstance(cmap, str)
    assert isinstance(top_words, pd.DataFrame)

#     print(cluster_num)
    wc = WordCloud(background_color="white", max_words=200, max_font_size=80, colormap=cmap);
    wordcloud = wc.generate(' '.join([word for word in top_words['Cluster #' + str(cluster_num)]]))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(label+ '_cluster_' + str(cluster_num), bbox_inches='tight')
    return None


def get_word_table(table, key, sim_key='similarity', show_sim = True):
    """""

    :param table: pre-processed confessions
    :type table: list
    :param key: pre-processed confessions
    :type key: list
    :returns: None
    """
    assert isinstance(table, list)
    assert isinstance(key, str)

    if show_sim == True:
        return pd.DataFrame(table, columns=[key, sim_key])
    else:
        return pd.DataFrame(table, columns=[key, sim_key])[key]
    
def compute_similarity_scores(confession, tags, model):
    """""
     
     :param confession: message to compute similarities to tags
     :type confession: str
     :param tags: keywords for comparison
     :type tags: list of str
     :returns: dictionary of array len(tags)
     """
    assert isinstance(confession, str)
    assert isinstance(tags, list)
    assert model != None
    similarity_scores = {}
    words = confession.split()

    for i, word in enumerate(words):
#         print(word)
    #         if str(word) not in model.wv.vocab:
    #             words = words.remove(word)
    #             print(confession)

        try:
            sim = [model.wv.similarity(str(word),tag) for tag in tags]
            similarity_scores[i] = sim
        except:
#             print('Error')
            similarity_scores[i] = list([0,0,0,0])
    
    return similarity_scores
    
def generate_tag_suggestions(scores, tags, threshold = 0.5):
    """""
     
     :param scores: similarities to  tags
     :type scores: dictionary
     :param tags: keywords for comparison
     :type tags: list of str
     :returns: dictionary of array len(tags)
     """
    assert isinstance(scores,dict)
    assert isinstance(tags,list)
    assert isinstance(threshold, float)
    
    results = []
    for key, value in scores.items():
        for i, sim in enumerate(value):
             if sim > threshold:
                results.append(tags[i])
                
    return results

def visualize_similarity_table(tags):
 """""
     
     :param tags: list of  tags
     :type tags: list
     """
    assert isinstance(tags,list)
    tables = [];
    for tag in tags:
        tables.append(get_word_table(model.wv.similar_by_word(tag), tag, show_sim=False))

    similarity_table = pd.concat(tables, axis=1)
    return similarity_table




