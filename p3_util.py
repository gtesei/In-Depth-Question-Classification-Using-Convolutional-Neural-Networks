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

from keras.layers import Reshape
from keras.layers.normalization import BatchNormalization

import re
from keras.preprocessing.text import Tokenizer

import warnings
from operator import itemgetter 
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.models import Sequential,Model
from keras.layers import CuDNNLSTM, Dense, Bidirectional, Input,Dropout
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda

############# FUNC #############

# https://www.kaggle.com/qqgeogor/keras-lstm-attention-glove840b-lb-0-043
class Attention(Layer):
    def __init__(self, step_dim=33,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


class Attention_CNN(Layer):
    def __init__(self, step_dim=2000,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention_CNN, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

def categorical_probas_to_classes(p):
    return np.argmax(p, axis=1)

def acquire_data(filename, model, maxim, index):
    loc_data = [] 
    questions = [] 
    for i in range(0, len(filename)):
        new_list = []
        current = filename[i].split()
        for c in current:
            if c in model:
                new_list.append(model[c])
        new_list = np.array(new_list)
        length = new_list.shape[0]
        sparser = np.zeros((maxim - length) * 300)
        new = np.reshape(new_list, (length * 300))
        vect = np.hstack((new, sparser))
        loc_data.append(vect)
        questions.append(filename[i])
    loc_data = np.array(loc_data)
    loc_targets = [index] * len(filename)
    return loc_data, np.array(loc_targets) , questions

def process_text(text):
    text = re.sub(r"\'s", " is ", text) 
    text = re.sub(r"\'s", " is ", text)
    text = re.sub(r"\'ve", " have ", text)
    return text



def make_dataset_3_cat(model_word_embed, 
                 EMBEDDING_DIM,
                 train_files = ['train_5500.txt'] , 
                 test_files = ['test_data.txt' , 'quora_test_set.txt'],
                 lower=False,
                 additional_question_list=None):
    
    all_files = train_files + test_files
    
    train_main_cat_list = [] 
    train_sub_cat_list = []
    train_text_list = []
    train_questions = []

    test_main_cat_list = [] 
    test_sub_cat_list = []
    test_text_list = []
    test_questions = []
    
    ##
    if additional_question_list is not None:
        additional_train_text_list = [] 
        for q in additional_question_list:
            #print(q)
            additional_train_text_list.append(text_to_word_sequence(process_text(q),lower=lower))
    
    ##
    for f in all_files: 
        lines = open(f, 'r' , encoding = "ISO-8859-1").read().splitlines()
        print(f," - size:",len(lines) , " - sample: ",lines[0])
        for i in range(len(lines)):
            tokens = lines[i].split()
            m_cat, _ = tokens[0].split(":")
            s_cat = tokens[0]
            tokens.pop(0)
            text = ' '.join(tokens)
            if f in train_files:
                train_main_cat_list.append(m_cat)
                train_sub_cat_list.append(s_cat)
                train_text_list.append(text_to_word_sequence(process_text(text),lower=lower))
                train_questions.append(lines[i])
            elif m_cat in train_main_cat_list and s_cat in train_sub_cat_list: 
                test_main_cat_list.append(m_cat)
                test_sub_cat_list.append(s_cat)
                test_text_list.append(text_to_word_sequence(process_text(text),lower=lower))
                test_questions.append(lines[i])
            else:
                print(">> removing: ",lines[i])
            
    assert len(train_main_cat_list) == len(train_sub_cat_list)
    assert len(train_main_cat_list) == len(train_text_list)
    assert len(train_main_cat_list) == len(train_questions)

    assert len(test_main_cat_list) == len(test_sub_cat_list)
    assert len(test_main_cat_list) == len(test_text_list)
    assert len(test_main_cat_list) == len(test_questions)

    MAX_SEQUENCE_LENGTH_TR = len(max(train_text_list,key=len))
    MAX_SEQUENCE_LENGTH_TS = len(max(test_text_list,key=len))
    MAX_SEQUENCE_LENGTH = max(MAX_SEQUENCE_LENGTH_TR,MAX_SEQUENCE_LENGTH_TS)
    print("MAX_SEQUENCE_LENGTH_TR:",MAX_SEQUENCE_LENGTH_TR)
    print("MAX_SEQUENCE_LENGTH_TS:",MAX_SEQUENCE_LENGTH_TS)
    print("--> MAX_SEQUENCE_LENGTH:",MAX_SEQUENCE_LENGTH)

    print(" Train - MAIN Categories:",len(set(train_main_cat_list))," - ",set(train_main_cat_list))
    print(" Train - Sub Categories:",len(set(train_sub_cat_list))," - ",set(train_sub_cat_list))

    print(" Test - MAIN Categories:",len(set(test_main_cat_list))," - ",set(test_main_cat_list))
    print(" Test - Sub Categories:",len(set(test_sub_cat_list))," - ",set(test_sub_cat_list))

    print(">> train_size:",len(train_main_cat_list))
    print(">> train sample:",train_main_cat_list[44] , train_sub_cat_list[44] , train_text_list[44] , train_questions[44])
    print(">> test_size:",len(test_main_cat_list))
    print(">> test sample:",test_main_cat_list[44] , test_sub_cat_list[44] , test_text_list[44] , test_questions[44])
    
    ##
    tokenizer = Tokenizer(num_words=None,char_level=False,lower=False)
    tokenizer.fit_on_texts(train_text_list + test_text_list) # + ... 
    sequences_train = tokenizer.texts_to_sequences(train_text_list) # ... train , test .. 
    sequences_test = tokenizer.texts_to_sequences(test_text_list) # ... train , test .. 
    data_train = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH,padding='post')
    data_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH,padding='post')
    if additional_question_list is not None:
        seq_data_additional_test = tokenizer.texts_to_sequences(additional_train_text_list) 
        data_additional = pad_sequences(seq_data_additional_test, maxlen=MAX_SEQUENCE_LENGTH,padding='post')
    else:
        data_additional = None 
        
    ##tokenizer.word_index
    print(">> data_train:",data_train.shape)
    print(">> train sample:",sequences_train[44] , data_train[44] , train_text_list[44] , train_questions[44])
    print(">> data_test:",data_test.shape)
    print(">> test sample:",sequences_test[44] , data_test[44] , test_text_list[44] , test_questions[44])
    
    #
    nb_words = len(tokenizer.word_index)+1
    embedding_matrix = np.zeros((nb_words, 300))
    if type(model_word_embed) is dict:
        # Glove
        for word, i in tokenizer.word_index.items():
            if word in model_word_embed.keys():
                #print('IN:',word)
                embedding_matrix[i] = model_word_embed[word]
            else:
                print('>>> OUT <<<:',word.encode('utf-8'))
    else:
        # Gensim 
        for word, i in tokenizer.word_index.items():
            if word in model_word_embed.vocab:
                #print('IN:',word)
                embedding_matrix[i] = model_word_embed.word_vec(word)
            else:
                print('>>> OUT <<<:',word.encode('utf-8'))
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    
    #
    label_encoder = LabelEncoder()
    label_encoder.fit(train_main_cat_list+test_main_cat_list)
    y_train_main_cat = label_encoder.transform(train_main_cat_list) 
    y_test_main_cat = label_encoder.transform(test_main_cat_list) 
    assert len(label_encoder.classes_) == len(set(train_main_cat_list))
    
    y_train_main_cat = np_utils.to_categorical(y_train_main_cat,num_classes=len(label_encoder.classes_))
    y_test_main_cat = np_utils.to_categorical(y_test_main_cat,num_classes=len(label_encoder.classes_))
    
    map_label_main = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    map_label_main = {v: k for k, v in map_label_main.items()}
    print("map_label_main::",map_label_main)
    
    #
    label_encoder = LabelEncoder()
    label_encoder.fit(train_sub_cat_list+test_sub_cat_list)
    y_train_sub_cat = label_encoder.transform(train_sub_cat_list) 
    y_test_sub_cat = label_encoder.transform(test_sub_cat_list) 
    
    print(len(label_encoder.classes_),len(set(train_sub_cat_list)))
    assert len(label_encoder.classes_) == len(set(train_sub_cat_list))
    
    y_train_sub_cat = np_utils.to_categorical(y_train_sub_cat,num_classes=len(label_encoder.classes_))
    y_test_sub_cat = np_utils.to_categorical(y_test_sub_cat,num_classes=len(label_encoder.classes_))
    
    map_label_sub = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    map_label_sub = {v: k for k, v in map_label_sub.items()}
    print("map_label_sub::",map_label_sub)
    #
    return data_train, y_train_main_cat, y_test_main_cat, data_test, y_train_sub_cat, y_test_sub_cat, embedding_matrix , train_questions, test_questions, map_label_main, map_label_sub, MAX_SEQUENCE_LENGTH, data_additional
    

def make_dataset_2_cat(model_word_embed, 
                 EMBEDDING_DIM,
                 train_files = ['train_5500.txt'] , 
                 test_files = ['test_data.txt' , 'quora_test_set.txt'],
                 lower=False):
    
    all_files = train_files + test_files
    
    train_main_cat_list = [] 
    train_sub_cat_list = []
    train_text_list = []
    train_questions = []

    test_main_cat_list = [] 
    test_sub_cat_list = []
    test_text_list = []
    test_questions = []
    
    ##
    for f in all_files: 
        lines = open(f, 'r' , encoding = "ISO-8859-1").read().splitlines()
        print(f," - size:",len(lines) , " - sample: ",lines[0])
        for i in range(len(lines)):
            tokens = lines[i].split()
            m_cat, _ = tokens[0].split(":")
            s_cat = tokens[0]
            tokens.pop(0)
            text = ' '.join(tokens)
            if f in train_files:
                train_main_cat_list.append(m_cat)
                train_sub_cat_list.append(s_cat)
                train_text_list.append(text_to_word_sequence(process_text(text),lower=lower))
                train_questions.append(lines[i])
            elif m_cat in train_main_cat_list and s_cat in train_sub_cat_list: 
                test_main_cat_list.append(m_cat)
                test_sub_cat_list.append(s_cat)
                test_text_list.append(text_to_word_sequence(process_text(text),lower=lower))
                test_questions.append(lines[i])
            else:
                print(">> removing: ",lines[i])
            
    assert len(train_main_cat_list) == len(train_sub_cat_list)
    assert len(train_main_cat_list) == len(train_text_list)
    assert len(train_main_cat_list) == len(train_questions)

    assert len(test_main_cat_list) == len(test_sub_cat_list)
    assert len(test_main_cat_list) == len(test_text_list)
    assert len(test_main_cat_list) == len(test_questions)

    MAX_SEQUENCE_LENGTH_TR = len(max(train_text_list,key=len))
    MAX_SEQUENCE_LENGTH_TS = len(max(test_text_list,key=len))
    MAX_SEQUENCE_LENGTH = max(MAX_SEQUENCE_LENGTH_TR,MAX_SEQUENCE_LENGTH_TS)
    print("MAX_SEQUENCE_LENGTH_TR:",MAX_SEQUENCE_LENGTH_TR)
    print("MAX_SEQUENCE_LENGTH_TS:",MAX_SEQUENCE_LENGTH_TS)
    print("--> MAX_SEQUENCE_LENGTH:",MAX_SEQUENCE_LENGTH)

    print(" Train - MAIN Categories:",len(set(train_main_cat_list))," - ",set(train_main_cat_list))
    print(" Train - Sub Categories:",len(set(train_sub_cat_list))," - ",set(train_sub_cat_list))

    print(" Test - MAIN Categories:",len(set(test_main_cat_list))," - ",set(test_main_cat_list))
    print(" Test - Sub Categories:",len(set(test_sub_cat_list))," - ",set(test_sub_cat_list))

    print(">> train_size:",len(train_main_cat_list))
    print(">> train sample:",train_main_cat_list[44] , train_sub_cat_list[44] , train_text_list[44] , train_questions[44])
    print(">> test_size:",len(test_main_cat_list))
    print(">> test sample:",test_main_cat_list[44] , test_sub_cat_list[44] , test_text_list[44] , test_questions[44])
    
    ##
    tokenizer = Tokenizer(num_words=None,char_level=False,lower=False)
    tokenizer.fit_on_texts(train_text_list + test_text_list) # + ... 
    sequences_train = tokenizer.texts_to_sequences(train_text_list) # ... train , test .. 
    sequences_test = tokenizer.texts_to_sequences(test_text_list) # ... train , test .. 
    data_train = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH,padding='post')
    data_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH,padding='post')
        
    ##tokenizer.word_index
    print(">> data_train:",data_train.shape)
    print(">> train sample:",sequences_train[44] , data_train[44] , train_text_list[44] , train_questions[44])
    print(">> data_test:",data_test.shape)
    print(">> test sample:",sequences_test[44] , data_test[44] , test_text_list[44] , test_questions[44])
    
    #
    nb_words = len(tokenizer.word_index)+1
    embedding_matrix = np.zeros((nb_words, 300))
    if type(model_word_embed) is dict:
        # Glove
        for word, i in tokenizer.word_index.items():
            if word in model_word_embed.keys():
                #print('IN:',word)
                embedding_matrix[i] = model_word_embed[word]
            else:
                print('>>> OUT <<<:',word.encode('utf-8'))
    else:
        # Gensim 
        for word, i in tokenizer.word_index.items():
            if word in model_word_embed.vocab:
                #print('IN:',word)
                embedding_matrix[i] = model_word_embed.word_vec(word)
            else:
                print('>>> OUT <<<:',word.encode('utf-8'))
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    
    #
    label_encoder = LabelEncoder()
    label_encoder.fit(train_main_cat_list+test_main_cat_list)
    y_train_main_cat = label_encoder.transform(train_main_cat_list) 
    y_test_main_cat = label_encoder.transform(test_main_cat_list) 
    assert len(label_encoder.classes_) == len(set(train_main_cat_list))
    
    y_train_main_cat = np_utils.to_categorical(y_train_main_cat,num_classes=len(label_encoder.classes_))
    y_test_main_cat = np_utils.to_categorical(y_test_main_cat,num_classes=len(label_encoder.classes_))
    
    map_label_main = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    map_label_main = {v: k for k, v in map_label_main.items()}
    print("map_label_main::",map_label_main)
    
    #
    label_encoder = LabelEncoder()
    label_encoder.fit(train_sub_cat_list+test_sub_cat_list)
    y_train_sub_cat = label_encoder.transform(train_sub_cat_list) 
    y_test_sub_cat = label_encoder.transform(test_sub_cat_list) 
    
    print(len(label_encoder.classes_),len(set(train_sub_cat_list)))
    assert len(label_encoder.classes_) == len(set(train_sub_cat_list))
    
    y_train_sub_cat = np_utils.to_categorical(y_train_sub_cat,num_classes=len(label_encoder.classes_))
    y_test_sub_cat = np_utils.to_categorical(y_test_sub_cat,num_classes=len(label_encoder.classes_))
    
    map_label_sub = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    map_label_sub = {v: k for k, v in map_label_sub.items()}
    print("map_label_sub::",map_label_sub)
    #
    return data_train, y_train_main_cat, y_test_main_cat, data_test, y_train_sub_cat, y_test_sub_cat, embedding_matrix , train_questions, test_questions, map_label_main, map_label_sub, MAX_SEQUENCE_LENGTH
    

def make_dataset(model_word_embed, 
                 word_embed_dim,
                 maxim, 
                 train_dir='train_5500/', # sub_categories 
                 test_dir='test_500/',  # sub_categories_test
                 sub_category='', 
                 #files = ['abbr.txt', 'desc.txt', 'enty.txt', 'hum.txt', 'loc.txt', 'num.txt']
                 ):
    train_data = []
    train_targets = []
    train_questions = [] 
    test_data = []
    test_targets = [] 
    test_questions = [] 
    files = os.listdir(train_dir + sub_category +'/') #['abbr.txt', 'desc.txt', 'enty.txt', 'hum.txt', 'loc.txt', 'num.txt']
    map_label = dict()
    for f in range(0, len(files)):
        if sub_category=='':
            map_label[f] = os.path.splitext(files[f])[0].upper()
        else:
            map_label[f] = sub_category.upper()+":"+os.path.splitext(files[f])[0]
        # train
        if sub_category=='':
            filename = open(train_dir + files[f], 'r' , encoding = "ISO-8859-1").read().splitlines()
        else:
            filename = open(train_dir + sub_category + '/' + files[f], 'r' , encoding = "ISO-8859-1").read().splitlines()
        loc_data, loc_targets, questions = acquire_data(filename, model_word_embed, maxim, f)
        print("TRAIN :: ",files[f],loc_data.shape, loc_targets.shape , len(questions))           
        train_data.append(loc_data)
        train_targets.append(loc_targets)
        train_questions.extend(questions)
        # test
        if sub_category=='':
            filename = test_dir + files[f]
        else:
            filename = test_dir + sub_category + '/' + files[f]
        if os.path.exists(filename):
            filename = open(filename).read().splitlines()            
            loc_data, loc_targets, questions = acquire_data(filename, model_word_embed, maxim, f)
            print("TEST :: ",files[f],loc_data.shape, loc_targets.shape , len(questions))                 
            test_data.append(loc_data)
            test_targets.append(loc_targets)
            test_questions.extend(questions)
    x_train = np.vstack(np.array(train_data))
    y_train = np.hstack(np.array(train_targets))
    x_test = np.vstack(np.array(test_data))
    y_test = np.hstack(np.array(test_targets))
    print("x_train:",x_train.shape," - y_train:",y_train.shape, "- train_questions:",len(train_questions))
    print("x_test:",x_test.shape," - y_test:",y_test.shape, "- test_questions:",len(test_questions))
    assert x_train.shape[0] == y_train.shape[0]
    assert x_train.shape[0] == len(train_questions)
    assert x_test.shape[0] == y_test.shape[0]
    assert x_test.shape[0] == len(test_questions)
    x_train = np.reshape(x_train, (x_train.shape[0], maxim, word_embed_dim, 1))
    x_test = np.reshape(x_test, (x_test.shape[0], maxim, word_embed_dim, 1))
    y_train = np_utils.to_categorical(y_train,num_classes=len(files))
    y_test = np_utils.to_categorical(y_test,num_classes=len(files))
    print("x_train:",x_train.shape," - y_train:",y_train.shape, "- train_questions:",len(train_questions))
    print("x_test:",x_test.shape," - y_test:",y_test.shape, "- test_questions:",len(test_questions))
    return x_train , y_train , x_test , y_test , train_questions , test_questions , map_label


def build_model(input_shape=(32,300,1),
                dropout_prob=0.5,
                n_classes=6):
    # model
    inputs = Input(shape= input_shape, dtype= 'float32')
    # 2-gram
    conv_1 = Conv2D(500, (2, 300), activation="relu") (inputs)
    max_pool_1 = MaxPooling2D(pool_size=(30, 1 ))(conv_1)
    # 3-gram
    conv_2 = Conv2D(500, (3, 300), activation="relu") (inputs)
    max_pool_2 = MaxPooling2D(pool_size=(29, 1 ))(conv_2)
    # 4-gram
    conv_3 = Conv2D(500, (4, 300), activation="relu") (inputs)
    max_pool_3 = MaxPooling2D(pool_size=(28, 1 ))(conv_3)
    # 5-gram
    conv_4 = Conv2D(500, (5, 300), activation="relu") (inputs)
    max_pool_4 = MaxPooling2D(pool_size=(27, 1))(conv_4)
    # concat 
    merged = concatenate([max_pool_1, max_pool_2, max_pool_3,max_pool_4])
    flatten = Flatten()(merged)
    # full-connect 
    full_conn = Dense(128, activation= 'tanh')(flatten)
    dropout_1 = Dropout(dropout_prob)(full_conn)
    full_conn_2 = Dense(64, activation= 'tanh')(dropout_1)
    dropout_2 = Dropout(dropout_prob)(full_conn_2)
    output = Dense(n_classes, activation= 'softmax')(dropout_2)
    model = Model(inputs, output)
    return model 

def build_model_tr_embed(MAX_SEQUENCE_LENGTH,
                embedding_matrix, 
                EMBEDDING_DIM, 
                dropout_prob=0.5,
                n_classes=6,
                tr_embed=True):

    embedding_layer = Embedding(embedding_matrix.shape[0],EMBEDDING_DIM,weights=[embedding_matrix],input_length=MAX_SEQUENCE_LENGTH,trainable=tr_embed)

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    embedded_sequences_rh = Reshape((MAX_SEQUENCE_LENGTH,EMBEDDING_DIM,1))(embedded_sequences)
    # 2-gram
    conv_1 = Conv2D(500, (2, EMBEDDING_DIM), activation="relu") (embedded_sequences_rh)
    max_pool_1 = MaxPooling2D(pool_size=(30, 1 ))(conv_1)
    # 3-gram
    conv_2 = Conv2D(500, (3, EMBEDDING_DIM), activation="relu") (embedded_sequences_rh)
    max_pool_2 = MaxPooling2D(pool_size=(29, 1 ))(conv_2)
    # 4-gram
    conv_3 = Conv2D(500, (4, EMBEDDING_DIM), activation="relu") (embedded_sequences_rh)
    max_pool_3 = MaxPooling2D(pool_size=(28, 1 ))(conv_3)
    # 5-gram
    conv_4 = Conv2D(500, (5, EMBEDDING_DIM), activation="relu") (embedded_sequences_rh)
    max_pool_4 = MaxPooling2D(pool_size=(27, 1))(conv_4)
    # concat 
    merged = concatenate([max_pool_1, max_pool_2, max_pool_3,max_pool_4])
    flatten = Flatten()(merged)
    # full-connect 
    full_conn = Dense(128, activation= 'tanh')(flatten)
    dropout_1 = Dropout(dropout_prob)(full_conn)
    full_conn_2 = Dense(64, activation= 'tanh')(dropout_1)
    dropout_2 = Dropout(dropout_prob)(full_conn_2)
    output = Dense(n_classes, activation= 'softmax')(dropout_2)
    model = Model(sequence_input, output)
    return model


def build_model_tr_embed_2_output(MAX_SEQUENCE_LENGTH,
                embedding_matrix, 
                EMBEDDING_DIM, 
                dropout_prob=0.5,
                n_classes_main=6,
                n_classes_sub=0,
                tr_embed=True):

    embedding_layer = Embedding(embedding_matrix.shape[0],EMBEDDING_DIM,weights=[embedding_matrix],input_length=MAX_SEQUENCE_LENGTH,trainable=tr_embed)

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    embedded_sequences_rh = Reshape((MAX_SEQUENCE_LENGTH,EMBEDDING_DIM,1))(embedded_sequences)

    ### -------------------------- MAIN CAT
    # 2-gram
    conv_1 = Conv2D(500, (2, EMBEDDING_DIM), activation="relu") (embedded_sequences_rh)
    max_pool_1 = MaxPooling2D(pool_size=(MAX_SEQUENCE_LENGTH-2, 1 ))(conv_1) # 30
    # 3-gram
    conv_2 = Conv2D(500, (3, EMBEDDING_DIM), activation="relu") (embedded_sequences_rh)
    max_pool_2 = MaxPooling2D(pool_size=(MAX_SEQUENCE_LENGTH-3, 1 ))(conv_2) # 29
    # 4-gram
    conv_3 = Conv2D(500, (4, EMBEDDING_DIM), activation="relu") (embedded_sequences_rh)
    max_pool_3 = MaxPooling2D(pool_size=(MAX_SEQUENCE_LENGTH-4, 1 ))(conv_3) # 28
    # 5-gram
    conv_4 = Conv2D(500, (5, EMBEDDING_DIM), activation="relu") (embedded_sequences_rh)
    max_pool_4 = MaxPooling2D(pool_size=(MAX_SEQUENCE_LENGTH-5, 1))(conv_4) # 27
    
    # concat 
    merged = concatenate([max_pool_1, max_pool_2, max_pool_3,max_pool_4])

    
    #merged = Reshape((1,-1))(merged)
    #flatten = Attention_CNN(1)(merged)
    flatten = Flatten()(merged)
    


    # full-connect -- MAIN  
    full_conn = Dense(128, activation= 'tanh')(flatten)
    dropout_1 = Dropout(dropout_prob)(full_conn)
    full_conn_2 = Dense(64, activation= 'tanh')(dropout_1)
    dropout_2 = Dropout(dropout_prob)(full_conn_2)
    output = Dense(n_classes_main, activation= 'softmax')(dropout_2)    

    #o2 = Reshape((1,1,6))(output)
    
    # concat 2  
    #merged_2 = concatenate([max_pool_1, max_pool_2, max_pool_3,max_pool_4])
    #flatten_2 = Flatten()(merged_2)

    # full-connect -- sub
    full_conn_sub = Dense(128, activation= 'tanh')(flatten)
    dropout_1_sub = Dropout(dropout_prob)(full_conn_sub)
    full_conn_2_sub = Dense(64, activation= 'tanh')(dropout_1_sub)
    dropout_2_sub = Dropout(dropout_prob)(full_conn_2_sub)
    output_sub = Dense(n_classes_sub, activation= 'softmax')(dropout_2_sub)
    
    model = Model(sequence_input, [output,output_sub])
    return model


def build_model_attention1_2_output(MAX_SEQUENCE_LENGTH,
                embedding_matrix, 
                EMBEDDING_DIM, 
                dropout_prob=0.5,
                n_classes_main=6,
                n_classes_sub=0,
                tr_embed=True):

    embedding_layer = Embedding(embedding_matrix.shape[0],EMBEDDING_DIM,weights=[embedding_matrix],input_length=MAX_SEQUENCE_LENGTH,trainable=tr_embed)

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    biLSTM_1 = Bidirectional(CuDNNLSTM(MAX_SEQUENCE_LENGTH, return_sequences=True))(embedded_sequences)
    biLSTM_2 = Bidirectional(CuDNNLSTM(MAX_SEQUENCE_LENGTH,return_sequences=False))(biLSTM_1)
    #att_1 = Attention(MAX_SEQUENCE_LENGTH)(biLSTM_2)
    att_1 = biLSTM_2

    # full-connect -- MAIN  
    full_conn = Dense(2*256, activation="relu")(att_1)
    output = Dense(n_classes_main, activation= 'softmax')(full_conn)

    # full-connect -- sub 
    full_conn_2 = Dense(2*256, activation="relu")(att_1)
    output_sub = Dense(n_classes_sub, activation= 'softmax')(full_conn_2)

    model = Model(sequence_input, [output,output_sub])
    return model


def build_model_attention2_2_output(MAX_SEQUENCE_LENGTH,
                embedding_matrix, 
                EMBEDDING_DIM, 
                dropout_prob=0.5,
                n_classes_main=6,
                n_classes_sub=0,
                tr_embed=True):

    def softmax(x, axis=1):
        ndim = K.ndim(x)
        if ndim == 2:
            return K.softmax(x)
        elif ndim > 2:
            e = K.exp(x - K.max(x, axis=axis, keepdims=True))
            s = K.sum(e, axis=axis, keepdims=True)
            return e / s
        else:
            raise ValueError('Cannot apply softmax to a tensor that is 1D')

    def one_step_attention(a):
        e = densor1(a)
        energies = densor2(e)
        alphas = activator(energies)
        context = dotor([alphas,a])
        return context

    embedding_layer = Embedding(embedding_matrix.shape[0],EMBEDDING_DIM,weights=[embedding_matrix],input_length=MAX_SEQUENCE_LENGTH,trainable=tr_embed)
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    biLSTM_1 = Bidirectional(CuDNNLSTM(MAX_SEQUENCE_LENGTH, return_sequences=True))(embedded_sequences)
    biLSTM_2 = Bidirectional(CuDNNLSTM(MAX_SEQUENCE_LENGTH,return_sequences=True))(biLSTM_1)

    ##
    densor1 = Dense(10, activation = "tanh")
    densor2 = Dense(1, activation = "relu")
    activator = Activation(softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
    dotor = Dot(axes = 1)

    ##
    context = one_step_attention(biLSTM_2)
    context = Flatten()(context)

    # full-connect -- MAIN  
    full_conn = Dense(MAX_SEQUENCE_LENGTH, activation="relu")(context)
    output = Dense(n_classes_main, activation=softmax)(full_conn)

    # full-connect -- sub 
    full_conn_2 = Dense(MAX_SEQUENCE_LENGTH, activation="relu")(context)
    output_sub = Dense(n_classes_sub, activation=softmax)(full_conn_2)

    model = Model(sequence_input, [output,output_sub])
    return model




def build_2_model(input_shape=(32,300,1),
                dropout_prob=0.5,
                n_classes=6):
    # model
    inputs = Input(shape= input_shape, dtype= 'float32')
    # 2-gram
    conv_1 = Conv2D(500, (2, 300), activation="relu") (inputs)
    max_pool_1 = MaxPooling2D(pool_size=(30, 1 ))(conv_1)
    # 3-gram
    conv_2 = Conv2D(500, (3, 300), activation="relu") (inputs)
    max_pool_2 = MaxPooling2D(pool_size=(29, 1 ))(conv_2)
    # 4-gram
    conv_3 = Conv2D(500, (4, 300), activation="relu") (inputs)
    max_pool_3 = MaxPooling2D(pool_size=(28, 1 ))(conv_3)
    # 5-gram
    conv_4 = Conv2D(500, (5, 300), activation="relu") (inputs)
    max_pool_4 = MaxPooling2D(pool_size=(27, 1))(conv_4)
    # concat 

    r = Reshape((32,300))(inputs)
    x = Bidirectional(LSTM(150, return_sequences=True, dropout=0.2, recurrent_dropout=0.2,trainable=True))(r)
    #x = MaxPooling2D()(x)
    #x = GlobalMaxPool1D()(x)
    x = Reshape((1,1,-1))(x)
   

    merged = concatenate([max_pool_1, max_pool_2, max_pool_3,max_pool_4 , x])
    flatten = Flatten()(merged)
    # full-connect 
    full_conn = Dense(128, activation= 'tanh')(flatten)
    dropout_1 = Dropout(dropout_prob)(full_conn)
    full_conn_2 = Dense(64, activation= 'tanh')(dropout_1)
    dropout_2 = Dropout(dropout_prob)(full_conn_2)
    output = Dense(n_classes, activation= 'softmax')(dropout_2)
    model = Model(inputs, output)
    return model 

def plot_learn_curve(history,do_plot=False):
    import matplotlib.pyplot as plt
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    if do_plot: 
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()
    else:
        return pd.DataFrame({'epochs': epochs , 'acc': acc , 'val_acc': val_acc, 'loss': loss, 'val_loss': val_loss},
                            index=epochs)


def test_accuracy(model,
                  x_test,
                  y_test,
                  test_questions,
                  map_label = {0:'Abbreviation', 1:'Description', 2:'Entity', 3:'Human', 4:'Location', 5:'Numeric'}, 
                  do_print=True ):
    _predictions = model.predict(x_test)
    _n_labels = _predictions.shape[1]
    predictions = categorical_probas_to_classes(_predictions)
    originals = categorical_probas_to_classes(y_test)
    lend = len(predictions) * 1.0
    acc = np.sum(predictions == originals)/lend
    if do_print:
        print("Test Accuracy:",acc)
        print(confusion_matrix(originals, predictions)) 
    #err_questions = list(itemgetter(*np.where(predictions != originals)[0].tolist())(test_questions))
    err_list = [] 
    pred_list = [] 
    truth_list = [] 
    for i in range(len(x_test)):
        if originals[i] != predictions[i]:
            err_list.append(test_questions[i])
            pred_list.append(map_label[predictions[i]])
            truth_list.append(map_label[originals[i]])
    err_dict = {'question': err_list , 'prediction': pred_list , 'truth': truth_list}
    for lab in range(_n_labels):
        probs = [] 
        for i in range(len(x_test)):
            if originals[i] != predictions[i]:
                probs.append(_predictions[i][lab])
        err_dict[str("prob_"+ map_label[lab])] = probs
    error_df = pd.DataFrame(err_dict)
    return acc , error_df

def test_accuracy2(model,
                  x_test,
                  y_test,
                  test_questions,
                  map_label = {0:'Abbreviation', 1:'Description', 2:'Entity', 3:'Human', 4:'Location', 5:'Numeric'}, 
                  do_print=True ):
    if len(map_label) < 10:
        _predictions = model.predict(x_test)[0] ## <<< main
    else:
        _predictions = model.predict(x_test)[1] ## <<< sub category 
    _n_labels = _predictions.shape[1]
    predictions = categorical_probas_to_classes(_predictions)
    originals = categorical_probas_to_classes(y_test)
    lend = len(predictions) * 1.0
    acc = np.sum(predictions == originals)/lend
    if do_print:
        print("Test Accuracy:",acc)
        print(confusion_matrix(originals, predictions)) 
    #err_questions = list(itemgetter(*np.where(predictions != originals)[0].tolist())(test_questions))
    err_list = [] 
    pred_list = [] 
    truth_list = [] 
    for i in range(len(x_test)):
        if originals[i] != predictions[i]:
            err_list.append(test_questions[i])
            pred_list.append(map_label[predictions[i]])
            truth_list.append(map_label[originals[i]])
    err_dict = {'question': err_list , 'prediction': pred_list , 'truth': truth_list}
    for lab in range(_n_labels):
        probs = [] 
        for i in range(len(x_test)):
            if originals[i] != predictions[i]:
                probs.append(_predictions[i][lab])
        err_dict[str("prob_"+ map_label[lab])] = probs
    error_df = pd.DataFrame(err_dict)
    return acc , error_df

def test_accuracy3(y_pred,
                  y_test,
                  test_questions,
                  map_label = {0:'Abbreviation', 1:'Description', 2:'Entity', 3:'Human', 4:'Location', 5:'Numeric'}, 
                  do_print=True ):
    if len(map_label) < 10:
        _predictions = y_pred[0] ## <<< main
    else:
        _predictions = y_pred[1] ## <<< sub category 
    _n_labels = _predictions.shape[1]
    predictions = categorical_probas_to_classes(_predictions)
    originals = categorical_probas_to_classes(y_test)
    lend = len(predictions) * 1.0
    acc = np.sum(predictions == originals)/lend
    if do_print:
        print("Test Accuracy:",acc)
        print(confusion_matrix(originals, predictions)) 
    #err_questions = list(itemgetter(*np.where(predictions != originals)[0].tolist())(test_questions))
    err_list = [] 
    pred_list = [] 
    truth_list = [] 
    for i in range(len(y_test)):
        if originals[i] != predictions[i]:
            err_list.append(test_questions[i])
            pred_list.append(map_label[predictions[i]])
            truth_list.append(map_label[originals[i]])
    err_dict = {'question': err_list , 'prediction': pred_list , 'truth': truth_list}
    for lab in range(_n_labels):
        probs = [] 
        for i in range(len(y_test)):
            if originals[i] != predictions[i]:
                probs.append(_predictions[i][lab])
        err_dict[str("prob_"+ map_label[lab])] = probs
    error_df = pd.DataFrame(err_dict)
    return acc , error_df
