#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import pickle
import os

import numpy as np
import gensim

from keras.models import Model, load_model

np.random.seed(2)
maxim = 32
w2v = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)  

def categorical_probas_to_classes(p):
    return np.argmax(p, axis=1)


def acquire_data(filename, model, maxim, index):
    loc_data = []
    for i in range(0, len(filename)):
        new_list = []
        current = filename[i].split()
        for c in current:
            if c in model:
                new_list.append(model[c])
        new_list = np.array(new_list)
        length = new_list.shape[0]
        #if length <= 32:
        sparser = np.zeros((maxim - length) * 300)
        new = np.reshape(new_list, (length * 300))
        vect = np.hstack((new, sparser))
        loc_data.append(vect)
        #else:
		#loc_data.append(new_list)
        #print i
    loc_data = np.array(loc_data)
    loc_targets = [index] * len(filename)
    return loc_data, np.array(loc_targets)


def classify_question(test_text):
    ## 
    data = []
    targets = []
    loc_data, loc_targets = acquire_data(test_text, w2v, maxim, -1)
    print loc_data.shape, loc_targets.shape           
    data.append(loc_data)
    data = np.array(data)
    data = np.vstack(data)
    x_test = data
    x_test = np.reshape(x_test, (x_test.shape[0], maxim, 300, 1))
    model = load_model('word2vec_main_model_10.h5')
    predictions = model.predict(x_test)
    prediction_labels = categorical_probas_to_classes(predictions)
    prediction_labels = prediction_labels.tolist()
    ## ['abbr.txt', 'desc.txt', 'enty.txt', 'hum.txt', 'loc.txt', 'num.txt']
    lab_dict = {0: "Abbreviation" , 1: "Description" , 2: "Entity" , 3: "Human" , 4: "Location" , 5: "Numeric"}
    prediction_labels = [lab_dict[i] for i in prediction_labels]
    print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", "Prediction:" , prediction_labels

###
#test_text = [' What is the full form of .com ? ', ' What does the abbreviation AIDS stand for ? ', " What does INRI stand for when used on Jesus ' cross ? "]

if __name__ == '__main__':
    test_text = [str(sys.argv[1])]
    #print ">>>>>>>>>>>>> Type EXIT to exit <<<<<<<<<<<<<<<<<"
    #for line in sys.stdin:
    #    #test_text = line.replace("\n","")
    #    test_text = line 
    #    if test_text == "EXIT":
    #        print("bye...")
    #        break
    #    else:
    print ">>>>>>>>>>>>>", test_text , " is ..."
    classify_question(test_text)






