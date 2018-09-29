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
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout, GRU

from keras.layers import Reshape

import warnings
from operator import itemgetter 

from p3_util import *

############# MAIN #############
if len(sys.argv) == 2:
	## sub-category 
	sub_category = sys.argv[1]
	PREFIX = "word2vec_"+sub_category+"_"
elif len(sys.argv) == 1:
	## main category 
	sub_category = ''
	PREFIX = "word2vec_main_"
else:
	raise Exception("See usage!!")

np.random.seed(2)
maxim = 32
word_embed_dim = 300
N_EPOCHS = 200

print(">> loading GoogleNews-vectors-negative300.bin ...")
w2v = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)  

print(">> making dataset / building model...")
x_train , y_train , x_test , y_test , train_questions , test_questions, map_label= make_dataset(w2v,word_embed_dim,sub_category=sub_category)
model = build_model(input_shape=(32,300,1))
model.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics= ['accuracy'])
model.summary()
earlystopper = EarlyStopping(patience=20, verbose=1,monitor='val_acc',mode='max')
checkpointer = ModelCheckpoint(PREFIX+'model.h5', verbose=1, save_best_only=True,monitor='val_acc',mode='max')
reduce_lr = ReduceLROnPlateau(factor=0.2, patience=5, min_lr=0.00001, verbose=1,monitor='val_acc',mode='max')

print(">> TRAINING ...")
results = model.fit(x_train, y_train,
                    validation_data=[x_test,y_test],
                    batch_size=50, epochs=N_EPOCHS,
                    callbacks=[earlystopper, checkpointer,reduce_lr])

learning_curve_df = plot_learn_curve(results,do_plot=False)
learning_curve_df.to_csv(PREFIX+'learning_curve.csv')

print(">> TEST ...")
model = load_model(PREFIX+'model.h5')
acc , error_df = test_accuracy(model,x_test,y_test,test_questions,map_label=map_label)
error_df.to_csv(PREFIX+'__val_acc_'+str(acc)+'__error_questions.csv')




