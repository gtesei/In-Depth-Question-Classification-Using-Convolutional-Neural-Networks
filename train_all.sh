#!/bin/bash

echo ">> TRAIN ALL (word2vec) <<"

echo "> training <main> ..."
#python layer_1_cnn_word2vec.py 10
python p3_cnn_w2v.py

echo "> training sub-category <abbreviatio>n .."
#python layer_2_cnn_word2vec.py abbr 10
python p3_cnn_w2v.py abbr 

echo "> training sub-category <entit>y .."
#python layer_2_cnn_word2vec.py enty 10
python p3_cnn_w2v.py enty

echo "> training sub-category <description> .."
#python layer_2_cnn_word2vec.py desc 10
python p3_cnn_w2v.py desc

echo "> training sub-category <human> .."
#python layer_2_cnn_word2vec.py hum 10
python p3_cnn_w2v.py hum

echo "> training sub-category <location> .."
#python layer_2_cnn_word2vec.py loc 10
python p3_cnn_w2v.py loc

echo "> training sub-category <numeric> .."
#python layer_2_cnn_word2vec.py num 10 
python p3_cnn_w2v.py num
