import os
import pandas as pd
from gensim import models, matutils
from allen_ai_funcs import *

os.chdir('C:\\Users\\Philipp.Deutsch\\Documents\\allen-ai')


stopwords = get_stopwords()

training   = prepare_data('Data\\training_set.tsv')
validation = prepare_data('Data\\validation_set.tsv')

training['train'] = 1
validation['train'] = 0

data = pd.concat([training, validation], ignore_index=True)

####################
questions_token = tokenize(data.question, stopwords)
answers_token   = tokenize(data.answer, stopwords)

#for q in range(len(validation_questions_token)):

data['match'] = 0
for i in data.loc[data.train==0].index:
    best_dist = -1
    best_q    = -1
    for j in data.loc[data.train==1].index:
        new_dist = matutils.cossim(questions_token[i], questions_token[j])
        if new_dist > best_dist:
            best_dist = new_dist
            best_q    = j
    data.loc[i, 'match'] = best_q
    if i > 10019:
        break
    

