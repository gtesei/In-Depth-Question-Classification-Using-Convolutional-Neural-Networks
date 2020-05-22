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







if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Index Elastic')
    parser.add_argument('-local_csv', help='data frame', dest='local_csv', type=str, default='hi_co.csv' , required=True)
    parser.add_argument('-field', help='question field', dest='field', type=str, default='Question' , required=True)
  
    args = parser.parse_args()
    
    # print parameters
    print('-' * 30)
    print('Parameters .')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)
    
    #####
    
    print(">>> reading local csv ... ")
    df = pd.read_csv(args.local_csv)
    question_list = df[args.field].tolist()
    
    
    print(">> making original dataset / building model...")
    data_train, y_train_main_cat, y_test_main_cat, data_test, y_train_sub_cat, y_test_sub_cat, embedding_matrix , train_questions, test_questions, map_label_main, map_label_sub, MAX_SEQUENCE_LENGTH,data_additional = make_dataset_3_cat(w2v,EMBEDDING_DIM, train_files = ['train_5500.txt'] , test_files = ['test_data.txt'],lower=False,additional_question_list=question_list) # , 'quora_test_set.txt'])
    
    
    #####
    PREFIX = "baseline_"
    np.random.seed(2)
    #MAX_SEQUENCE_LENGTH = 32
    EMBEDDING_DIM = 300
    
    model = load_model(PREFIX+'model.h5')
    print(">> loading GoogleNews-vectors-negative300.bin ...")
    w2v = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)  
    
    ####
    print(">> TEST ...")
    print("> Sub category:")
    acc_sub , error_df_sub = test_accuracy2(model,data_test,y_test_sub_cat,test_questions,map_label=map_label_sub)
    print("> Main category:")
    acc_main , error_df_main = test_accuracy2(model,data_test,y_test_main_cat,test_questions,map_label=map_label_main)
    error_df_sub.to_csv(PREFIX+'__val_acc_'+str(acc_sub)+'__error_questions.csv')
    ####
    
    main_pred , sub_pred = model.predict(question_list)
    main_pred_list , sub_pred_list = [] , [] 
    for i in range(len(question_list)):
        main_pred_list.append(map_label_main[predictions[i]])
        sub_pred_list.append(map_label_sub[predictions[i]])
        
    ##
    df['TRAC_main_pred'] = main_pred_list
    df['TRAC_sub_pred'] = sub_pred_list
    #
    df.to_csv('csv_with_prediction.csv',index=False)
    
    
    
