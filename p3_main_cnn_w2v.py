import sys
import pickle
import os

import numpy as np
import gensim
import pandas as pd 

from sklearn.metrics import confusion_matrix

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
if len(sys.argv) == 2:
	if sys.argv[1] == "main":
                if not os.path.exists('results/main_word2vec'):
                        os.makedirs('results/main_word2vec')
                PREFIX = "results/main_word2vec/word2vec_main_"
                print(">> MAIN-category")
                IS_MAIN_CAT = True 
	elif sys.argv[1] == "sub":
                if not os.path.exists('results/flat_word2vec'):
                        os.makedirs(os.path.dirname('results/flat_word2vec'),exist_ok=True)
                PREFIX = "results/flat_word2vec/word2vec_flat_"
                print(">> Sub-category")
                IS_MAIN_CAT = False  
	else:
	        raise Exception("Usage: "+str(os.path.basename(__file__))+" main|sub")
else:
    raise Exception("Usage: "+str(os.path.basename(__file__))+" main|sub")

np.random.seed(2)
MAX_SEQUENCE_LENGTH = 32
EMBEDDING_DIM = 300
N_EPOCHS = 200


print(">> loading GoogleNews-vectors-negative300.bin ...")
w2v = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)  

print(">> making dataset / building model...")
data_train, y_train_main_cat, y_test_main_cat, data_test, y_train_sub_cat, y_test_sub_cat, embedding_matrix , train_questions, test_questions, map_label_main, map_label_sub,MAX_SEQUENCE_LENGTH = make_dataset_2_cat(w2v,EMBEDDING_DIM, train_files = ['train_5500.txt'] , test_files = ['test_data.txt'],lower=False) # , 'quora_test_set.txt'])

if IS_MAIN_CAT:
	labels_train = y_train_main_cat
	labels_test = y_test_main_cat
	_map_label = map_label_main
else:
	labels_train = y_train_sub_cat
	labels_test = y_test_sub_cat
	_map_label = map_label_sub

model = build_model_tr_embed(MAX_SEQUENCE_LENGTH,embedding_matrix, EMBEDDING_DIM, dropout_prob=0.5,n_classes=len(_map_label),tr_embed=False)
model.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics= ['accuracy'])
model.summary()
earlystopper = EarlyStopping(patience=20, verbose=1,monitor='val_acc',mode='max')
checkpointer = ModelCheckpoint(PREFIX+'model.h5', verbose=1, save_best_only=True,monitor='val_acc',mode='max')
reduce_lr = ReduceLROnPlateau(factor=0.2, patience=5, min_lr=0.00001, verbose=1,monitor='val_acc',mode='max')

print(">> TRAINING ...")
results = model.fit(data_train, labels_train,
                    validation_data=[data_test,labels_test],
                    batch_size=50, epochs=N_EPOCHS,
                    callbacks=[earlystopper, checkpointer,reduce_lr])

learning_curve_df = plot_learn_curve(results,do_plot=False)
learning_curve_df.to_csv(PREFIX+'learning_curve.csv')

print(">> TEST ...")
model = load_model(PREFIX+'model.h5')
acc , error_df = test_accuracy(model,data_test,labels_test,test_questions,map_label=_map_label)
error_df.to_csv(PREFIX+'__val_acc_'+str(acc)+'__error_questions.csv')




