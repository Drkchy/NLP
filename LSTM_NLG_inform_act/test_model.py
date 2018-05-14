# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import argparse

# Import useful packages
import re
import io
import os
import pickle
import numpy as np
import pandas as pd
import copy
import random
from keras.models import model_from_json
from gensim.models import KeyedVectors
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Embedding, LSTM, RepeatVector, Dense, Activation, Input, Flatten, Reshape, Permute, Lambda
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import one_hot
from keras.utils import to_categorical

# fix random seed for reproducibility
np.random.seed(7)
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
import sys
import json



def delex_test(mrs, update_data_source=False, specific_slots=None, split=True):
    if specific_slots is not None:
        delex_slots = specific_slots
    else:
        delex_slots = ['name', 'food', 'near']

    for x, mr in enumerate(mrs):
        for slot_value in mr.split(','):
            sep_idx = slot_value.find('[')
            # parse the slot
            slot = slot_value[:sep_idx].strip()
            if slot in delex_slots:
                value = slot_value[sep_idx + 1:-1].strip()
                mr = mr.replace(value, '&slot_val_{0}&'.format(slot))
        if update_data_source:
            mrs[x] = mr


def clean_mrs(list):
    new_list = []
    
    for mr in list:
        row_list = []
        for word in mr.split(','):
            sep_idx = word.find('[')
            slot = word[:sep_idx].strip().lower()
            
            row_list.extend(slot_word for slot_word in slot.split())
        
            value = word[sep_idx+1:-1].strip().lower()
            
            row_list.extend(value_word for value_word in value.split())
            
#            for value_word in value.split():
#                if value_word not in mr_words:
#                    mr_words.add(value_word.lower())
#            for slot_value in slot.split():
#                if slot_value not in mr_words:
#                    mr_words.add(slot_value.lower())
                  
        new_list.append(row_list)
    return new_list


def load_embedding_model(path_to_model):  
    return KeyedVectors.load_word2vec_format(path_to_model, binary=False)


def add_padding(seq, padding_vec, max_seq_len):
    diff = max_seq_len - len(seq)
    if diff > 0:
        # pad short sequences
        return seq + [padding_vec for i in range(diff)]
    else:
        # truncate long sequences
        return seq[:max_seq_len]

def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dataset', help='pathname_to_test_dataset', required=True)
    parser.add_argument('--model_file', help='pathname_to_model', required=True)
    parser.add_argument('--output_test_file', help='pathname_to_results_testfile', required=True)

    opts = parser.parse_args()
    
    path_to_testset = opts.test_dataset
    path_to_model = opts.model_file
    path_to_new_file = opts.output_test_file

    path_to_embedding_model = os.path.join(path_to_model,'embedding_model.bin') 
    path_to_embeddings = os.path.join(path_to_model,'embeddings.npy') 
    path_to_dict = os.path.join(path_to_model,'dict.pkl') 
    path_to_vocab = os.path.join(path_to_model,'vocab.json')
    
    testset = pd.read_csv(path_to_testset, header=0, encoding='utf-8') 
    testset_mrs = testset.MR.tolist()
    
    # =============================================================================
    #     Load model 
    # =============================================================================
    
    print('Load saved model')
    json_file = open(os.path.join(path_to_model,'model.json'),'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    loaded_model.load_weights(os.path.join(path_to_model,"model.h5"))
    print('Weights loaded from disk')
    
    # load idx2word dict
    target_token_index = load(path_to_dict)
    print(target_token_index)
    target_index_token = {idx: word for word, idx in target_token_index.items()}

    
    
    
    delex_test(testset_mrs, update_data_source=True)
    x_test_seq = clean_mrs(testset_mrs)

    
    # Load embedding
    embedding_model = load_embedding_model(path_to_embedding_model)
    weights = embedding_model.syn0 
        
    padding_vec = np.zeros(embedding_model.syn0.shape[1]) 

    X_test=[]
    
    max_encoder_seq_length = 23
    
    for mrs in x_test_seq:
        new_mr=[]
        for word in mrs:
            new_mr.append(embedding_model[word])
        padded_new_mr = add_padding(new_mr, padding_vec, max_encoder_seq_length)
        X_test.append(padded_new_mr)
    
    target_index_token = {idx: word for word, idx in target_token_index.items()}



    # -- BATCH PREDICTION --
    results = []
    prediction_distr = loaded_model.predict(np.array(X_test))
    predictions = np.argmax(prediction_distr, axis=1)
    
    for i, prediction in enumerate(predictions):
        #print(prediction)
        utterance = ' '.join([target_index_token[idx] for idx in prediction if idx > 0])
        results.append(utterance)
    
    print(results)