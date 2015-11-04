import re
import csv
import string
from collections import defaultdict
from gensim import corpora
import pandas as pd


def prepare_data(infile):
    data = pd.read_csv(infile, sep="\t")
    data = pd.melt(data, 
                       id_vars = ['id', 'question', 'correctAnswer'],
                       value_vars = ['answerA', 'answerB', 'answerC', 'answerD']
                       )
    data.loc[data.variable == 'answerA', 'variable'] = 'A'
    data.loc[data.variable == 'answerB', 'variable'] = 'B'
    data.loc[data.variable == 'answerC', 'variable'] = 'C'
    data.loc[data.variable == 'answerD', 'variable'] = 'D'

    data['correct'] = 0
    data.loc[data.variable == data.correctAnswer, 'correct'] = 1
    data.drop(['correctAnswer', 'variable'], inplace=True, axis=1)
    data.rename(columns={'value' : 'answer'}, inplace=True)

    return data

def get_stopwords():
    with open('Data\\stop-word-list.csv', 'r') as csvfile:
        swreader = csv.reader(csvfile, delimiter=',')
        for row in swreader:
            stopwords = row
    stopwords = [s.strip() for s in stopwords]
    return stopwords

def remove_punctuation(s):
    return re.sub('[%s]' % re.escape(string.punctuation), '', s)

def tokenize(documents, stopwords):
    documents = [remove_punctuation(s) for s in documents]
    texts = [[word for word in document.lower().split() if word not in stopwords]
             for document in documents]
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [[token for token in text if frequency[token] > 1]
             for text in texts]

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    return corpus
