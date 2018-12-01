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

######################
def process_glove(we_fn='glove.6B.300d.txt'):
    print('>> Glove ...')
    embeddings_index = {}
    f = open(os.path.join('data', we_fn))
    for line in f:
        values = line.split(' ')
        word = values[0] #print("values:",values)
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

def get_w2v(name):
    if name == "goog":
        print(">> loading GoogleNews-vectors-negative300.bin ...")
        w2v = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)  
        #w2v = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300-hard-debiased.bin', binary=True)  
    elif name == "glove":
        print(">> loading glove.840B.300d.txt ...")
        w2v = process_glove('/home/ubuntu/var/quora-insincere-questions-classification/data/glove.840B.300d/glove.840B.300d.txt')
    elif name == "fasttext":
        print(">> loading wiki-news-300d-1M.vec ...")
        w2v = gensim.models.KeyedVectors.load_word2vec_format('/home/ubuntu/var/quora-insincere-questions-classification/data/wiki-news-300d-1M/wiki-news-300d-1M.vec')##fasttext
    else:
        raise Exception("w2v not supported:"+str(name))
    return w2v


############# MAIN #############
#os.makedirs(os.path.dirname('results/double/a.csv'))
PREFIX = "results/double/word2vec_double_"
print(">> Double")

np.random.seed(2)
EMBEDDING_DIM = 300
N_EPOCHS = 200
W2V_LIST = = ["goog","glove","fasttext"] 
FILE_OUT = "ensemb_results.txt"

preds = [] 
for wt in W2V_LIST:
    K.clear_session()
    print(">>>>>>>>>>>>>>>>>>>>>>>> ",wt,"<<<<<<<<<<<<<<<<<<<<<<<<<<,")
    w2v = get_w2v(wt)
    print(">> making dataset / building model...")
    data_train, y_train_main_cat, y_test_main_cat, data_test, y_train_sub_cat, y_test_sub_cat, embedding_matrix , train_questions, test_questions, map_label_main, map_label_sub, MAX_SEQUENCE_LENGTH = make_dataset_2_cat(w2v,EMBEDDING_DIM, train_files = ['train_5500.txt'] , test_files = ['test_data.txt'],lower=False) # , 'quora_test_set.txt'])
    model = build_model_tr_embed_2_output(MAX_SEQUENCE_LENGTH,embedding_matrix, EMBEDDING_DIM, dropout_prob=0.5,n_classes_main=len(map_label_main),n_classes_sub=len(map_label_sub),tr_embed=False)
    model.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics= ['accuracy'],loss_weights=[0.5, 0.5])
    model.summary()
    earlystopper = EarlyStopping(patience=20, verbose=1,monitor='val_dense_6_acc',mode='max')
    checkpointer = ModelCheckpoint(PREFIX+wt+'_model.h5', verbose=1, save_best_only=True,monitor='val_dense_6_acc',mode='max')
    reduce_lr = ReduceLROnPlateau(factor=0.2, patience=5, min_lr=0.00001, verbose=1,monitor='val_dense_6_acc',mode='max')
    print(">> TRAINING ",i,"/",REPEAT,"...")
    results = model.fit(data_train, [y_train_main_cat,y_train_sub_cat],
                    validation_data=[data_test,[y_test_main_cat,y_test_sub_cat]],
                    batch_size=66, epochs=N_EPOCHS,
                    callbacks=[earlystopper, checkpointer,reduce_lr])
    #learning_curve_df = plot_learn_curve(results,do_plot=False)
    #learning_curve_df.to_csv(PREFIX+'learning_curve.csv')
    print(">> TEST ...")
    model = load_model(PREFIX+wt+'_model.h5')
    pred = model.predict(data_test)
    preds.append(pred)

## 
print(">>>> Model Averaging ... ")
(pred[0] + pred[0]) /2

pmain, psub = None, None 
for p in preds:
    if pmain = None:
        pmain = p[0]
        psub = p[1]
    else:
        pmain = pmain + p[0]
        psub = psub + p[1]

pmain = pmain / len(preds) 
psub = psub / len(preds) 
    
print(">>> Sub category:")
acc_sub , error_df_sub = test_accuracy2(model,data_test,y_test_sub_cat,test_questions,map_label=map_label_sub)
print(">> acc_sub:",acc_sub)
print(">>> Main category:")
acc_main , error_df_main = test_accuracy2(model,data_test,y_test_main_cat,test_questions,map_label=map_label_main)
print(">> acc_main:",acc_main)
# error_df_sub.to_csv(PREFIX+'__val_acc_'+str(acc_sub)+'__error_questions.csv')


   
    




