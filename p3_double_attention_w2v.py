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

from sklearn.preprocessing import LabelEncoder

import re
from keras.preprocessing.text import Tokenizer

import warnings
from operator import itemgetter 
import os

from p3_util import *

############# MAIN #############
#os.makedirs(os.path.dirname('results/double/a.csv'))
PREFIX = "results/double/word2vec_double_"
print(">> Double")

np.random.seed(2)
#MAX_SEQUENCE_LENGTH = 32
EMBEDDING_DIM = 300
N_EPOCHS = 200
REPEAT = 1
FILE_OUT = "attention_results.txt"

print(">> loading GoogleNews-vectors-negative300-hard-debiased.bin ...")
w2v = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)  
#w2v = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300-hard-debiased.bin', binary=True)  

print(">> making dataset / building model...")
data_train, y_train_main_cat, y_test_main_cat, data_test, y_train_sub_cat, y_test_sub_cat, embedding_matrix , train_questions, test_questions, map_label_main, map_label_sub, MAX_SEQUENCE_LENGTH = make_dataset_2_cat(w2v,EMBEDDING_DIM, train_files = ['train_5500.txt'] , test_files = ['test_data.txt'],lower=False) # , 'quora_test_set.txt'])

acc_mains , acc_subs = [] ,[]
for i in range(REPEAT):
    K.clear_session()
    model = build_model_attention1_2_output(MAX_SEQUENCE_LENGTH,embedding_matrix, EMBEDDING_DIM, dropout_prob=0.5,n_classes_main=len(map_label_main),n_classes_sub=len(map_label_sub),tr_embed=False)
    model.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics= ['accuracy'],loss_weights=[0.5, 0.5])
    model.summary()
    ## 1 
    earlystopper = EarlyStopping(patience=20, verbose=1,monitor='val_dense_4_acc',mode='max')
    checkpointer = ModelCheckpoint(PREFIX+'model.h5', verbose=1, save_best_only=True,monitor='val_dense_4_acc',mode='max')
    reduce_lr = ReduceLROnPlateau(factor=0.2, patience=5, min_lr=0.00001, verbose=1,monitor='val_dense_4_acc',mode='max')
    ## 2
    #earlystopper = EarlyStopping(patience=20, verbose=1,monitor='val_dense_6_acc',mode='max')
    #checkpointer = ModelCheckpoint(PREFIX+'model.h5', verbose=1, save_best_only=True,monitor='val_dense_6_acc',mode='max')
    #reduce_lr = ReduceLROnPlateau(factor=0.2, patience=5, min_lr=0.00001, verbose=1,monitor='val_dense_6_acc',mode='max')

    print(">> TRAINING ",i,"/",REPEAT,"...")
    results = model.fit(data_train, [y_train_main_cat,y_train_sub_cat],
                    validation_data=[data_test,[y_test_main_cat,y_test_sub_cat]],
                    batch_size=50, epochs=N_EPOCHS,
                    callbacks=[earlystopper, checkpointer,reduce_lr])

    #learning_curve_df = plot_learn_curve(results,do_plot=False)
    #learning_curve_df.to_csv(PREFIX+'learning_curve.csv')
    print(">> TEST ...")
    model = load_model(PREFIX+'model.h5',custom_objects={"Attention": Attention(MAX_SEQUENCE_LENGTH)})
    print("> Sub category:")
    acc_sub , error_df_sub = test_accuracy2(model,data_test,y_test_sub_cat,test_questions,map_label=map_label_sub)
    print("> Main category:")
    acc_main , error_df_main = test_accuracy2(model,data_test,y_test_main_cat,test_questions,map_label=map_label_main)
    #error_df_sub.to_csv(PREFIX+'__val_acc_'+str(acc_sub)+'__error_questions.csv')
    acc_mains.append(acc_main)
    acc_subs.append(acc_sub)
    print("**** acc_mains ***",file=open(FILE_OUT, "w"))
    print(acc_mains,file=open(FILE_OUT, "a"))
    print("acc_mains avg:",np.array(acc_mains).mean(),file=open(FILE_OUT, "a"))
    print("**** acc_subs ***",file=open(FILE_OUT, "a"))
    print(acc_subs,file=open(FILE_OUT, "a"))
    print("acc_subs avg:",np.array(acc_subs).mean(),file=open(FILE_OUT, "a"))
    print("End.",file=open(FILE_OUT, "a"))




