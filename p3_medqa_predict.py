#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 11:26:26 2020

@author: ag62216
"""


import sys
import pickle
import os

import numpy as np
import gensim
import pandas as pd 

from sklearn.metrics import confusion_matrix
from keras import backend as K
from keras.models import Model, load_model
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Activation, Flatten, Input, Dense, Dropout
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

import re
from keras.preprocessing.text import Tokenizer

import warnings
from operator import itemgetter 
import os

from p3_util import *

#####
PREFIX = "baseline_"
np.random.seed(2)
#MAX_SEQUENCE_LENGTH = 32
EMBEDDING_DIM = 300

model = load_model(PREFIX+'model.h5')
print(">> loading GoogleNews-vectors-negative300.bin ...")
w2v = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)  
#w2v = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300-hard-debiased.bin', binary=True)  
#w2v = process_glove('/home/ubuntu/var/quora-insincere-questions-classification/data/glove.840B.300d/glove.840B.300d.txt')
#w2v = gensim.models.KeyedVectors.load_word2vec_format('/home/ubuntu/var/quora-insincere-questions-classification/data/wiki-news-300d-1M/wiki-news-300d-1M.vec')##fasttext  

print(">> making dataset / building model...")
data_train, y_train_main_cat, y_test_main_cat, data_test, y_train_sub_cat, y_test_sub_cat, embedding_matrix , train_questions, test_questions, map_label_main, map_label_sub, MAX_SEQUENCE_LENGTH = make_dataset_2_cat(w2v,EMBEDDING_DIM, train_files = ['train_5500.txt'] , test_files = ['test_data.txt'],lower=False) # , 'quora_test_set.txt'])

####
print(">> TEST ...")
print("> Sub category:")
acc_sub , error_df_sub = test_accuracy2(model,data_test,y_test_sub_cat,test_questions,map_label=map_label_sub)
print("> Main category:")
acc_main , error_df_main = test_accuracy2(model,data_test,y_test_main_cat,test_questions,map_label=map_label_main)
error_df_sub.to_csv(PREFIX+'__val_acc_'+str(acc_sub)+'__error_questions.csv')

