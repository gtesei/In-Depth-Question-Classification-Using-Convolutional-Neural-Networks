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


import re
from keras.preprocessing.text import Tokenizer

import warnings
from operator import itemgetter 

############# FUNC #############
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



def make_dataset_2_cat(model_word_embed, 
                 EMBEDDING_DIM,
                 MAX_SEQUENCE_LENGTH, 
                 train_files = ['train_5500.txt'] , 
                 test_files = ['test_data.txt' , 'quora_test_set.txt']):
    
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
                train_text_list.append(text_to_word_sequence(process_text(text),lower=False))
                train_questions.append(lines[i])
            elif m_cat in train_main_cat_list and s_cat in train_sub_cat_list: 
                test_main_cat_list.append(m_cat)
                test_sub_cat_list.append(s_cat)
                test_text_list.append(text_to_word_sequence(process_text(text),lower=False))
                test_questions.append(lines[i])
            else:
                print(">> removing: ",lines[i])
            
    assert len(train_main_cat_list) == len(train_sub_cat_list)
    assert len(train_main_cat_list) == len(train_text_list)
    assert len(train_main_cat_list) == len(train_questions)

    assert len(test_main_cat_list) == len(test_sub_cat_list)
    assert len(test_main_cat_list) == len(test_text_list)
    assert len(test_main_cat_list) == len(test_questions)

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
    for word, i in tokenizer.word_index.items():
        if word in model_word_embed.vocab:
            #print('IN:',word)
            embedding_matrix[i] = model_word_embed.word_vec(word)
        else:
            print('>>> OUT <<<:',word)
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
    return data_train, y_train_main_cat, y_test_main_cat, data_test, y_train_sub_cat, y_test_sub_cat, embedding_matrix , train_questions, test_questions, map_label_main, map_label_sub    
    

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
