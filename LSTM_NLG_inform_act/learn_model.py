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
from nltk.probability import FreqDist

import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Embedding, LSTM, RepeatVector, Dense, Activation, Input, Flatten, Reshape, Permute, Lambda
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import one_hot
from keras.utils import to_categorical
    
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.merge import multiply, concatenate
from keras.callbacks import ModelCheckpoint
    
    

# fix random seed for reproducibility
np.random.seed(7)
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors

import sys
import json
import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess



def preprocess(ref):
    to_remove = '!"#$%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n'
    ref = ref.replace('. ', ' . ')
    if ref[-1] == '.':
        ref = ref[:-1] + ' ' + ref[-1]
    return text_to_word_sequence(ref, filters=to_remove)   

def delexicalisation(mrs, sentences, update_data_source=False, specific_slots=None, split=True):
    if specific_slots is not None:
        delex_slots = specific_slots
    else:
        delex_slots = ['name', 'food', 'near']

    for x, mr in enumerate(mrs):
        if split:
            sentence = ' '.join(sentences[x])
        else:
            sentence = sentences[x].lower()
        for slot_value in mr.split(','):
            sep_idx = slot_value.find('[')
            # parse the slot
            slot = slot_value[:sep_idx].strip()
            if slot in delex_slots:
                value = slot_value[sep_idx + 1:-1].strip()
                sentence = sentence.replace(value.lower(), '&slot_val_{0}&'.format(slot))
                mr = mr.replace(value, '&slot_val_{0}&'.format(slot))
        if update_data_source:
            if split:
                sentences[x] = sentence.split()
            else:
                sentences[x] = sentence
            mrs[x] = mr
        if not split:
            return sentence

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
            
            for value_word in value.split():
                if value_word not in mr_words:
                    mr_words.add(value_word.lower())
            for slot_value in slot.split():
                if slot_value not in mr_words:
                    mr_words.add(slot_value.lower())
                  
        new_list.append(row_list)
    return new_list
        

def create_embeddings(sentences, path_to_embeddings, path_to_vocab, path_to_embedding_model, **params):
    model = Word2Vec(sentences, **params)
    model.wv.save_word2vec_format(path_to_embedding_model)
    
    weights = model.wv.syn0
    np.save(open(path_to_embeddings, 'wb'), weights)

    vocab = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    #print(vocab)
    with open(path_to_vocab, 'w') as f:
        f.write(json.dumps(vocab))


def load_embedding_model(path_to_embedding_model):  
    return KeyedVectors.load_word2vec_format(path_to_embedding_model, binary=False)


def add_padding(seq, padding_vec, max_seq_len):
    diff = max_seq_len - len(seq)
    if diff > 0:
        # pad short sequences
        return seq + [padding_vec for i in range(diff)]
    else:
        # truncate long sequences
        return seq[:max_seq_len]

def save(dict,path):
    pickle.dump(dict, open(path, 'wb'))
  

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset', help='pathname_to_train_dataset', required=True)
    parser.add_argument('--output_model_file', help='pathname_to_model_file', required=True)

    opts = parser.parse_args()
    
    
    path_to_trainset = opts.train_dataset
    path_to_model = opts.output_model_file

    path_to_embedding_model = os.path.join(path_to_model,'embedding_model.bin') 
    path_to_embeddings = os.path.join(path_to_model,'embeddings.npy') 
    path_to_dict = os.path.join(path_to_model,'dict.pkl') 
    path_to_vocab = os.path.join(path_to_model,'vocab.json')
    
    # load csv file
    print('load csv file')
    trainset = pd.read_csv(path_to_trainset, header=0, encoding='utf-8') 
    
    # randomly select 5% of training set to fit and validate the models 
    # 95% of the remaining graph is used to create the graph features
    to_keep = random.sample(range(len(trainset)), 
                            k=int(round(len(trainset)*0.05)))
    trainset = trainset.iloc[to_keep]
    df_rest = trainset.loc[~trainset.index.isin(to_keep)]
    
    
    trainset_mrs = trainset.mr.tolist()
    trainset_mrs_v0 = trainset.mr.tolist() #save an original version
    trainset_refs = trainset.ref.tolist()
    
    # build 2 voc dict
    mr_words = set()
    ref_words = set()
    
    # =============================================================================
    # Preprocess mrs & ref : trainset & testset
    # =============================================================================
    
    ## Clean REFS : remove punctuations and make a list 
    print('Clean refs')
    y_train_seq = [preprocess(_.lower()) for _ in trainset_refs]

    # Apply delexicalisation on MRS
    print('Apply delexicalisation')
    delexicalisation(trainset_mrs, y_train_seq, update_data_source=True)
    
    # Build a list for vocab with train and dev set (REFS)
    for ref in y_train_seq:
        for word in ref:
            if word not in ref_words:
                ref_words.add(word)
    
    ## Clean MRs : remove brackets and make a list 
    print('Clean MRs')
    
    # list of cleaned mrs, delexicalised
    x_train_seq = clean_mrs(trainset_mrs)
    
    # list of cleaned mrs, non delexicalised
    x_train_seq_v0 = clean_mrs(trainset_mrs_v0)
    
    
    
    # =============================================================================
    # Build  dicts and voc sizes
    # =============================================================================
    
    print('Build  dicts and voc sizes')
    mr_words = list(mr_words)
    ref_words = list(ref_words)

    mr_words = sorted(mr_words)
    ref_words = sorted(ref_words)
    
    input_token_index = dict(
        [(wd, i) for i, wd in enumerate(mr_words)])
    
    target_token_index = dict(
        [(wd, i) for i, wd in enumerate(ref_words)])
    
    save(target_token_index,path_to_dict)
    
    size_voc_mrs = len(input_token_index)
    size_voc_ref = len(target_token_index)
        
    # Following Keras_team template (lstm_seq2seq.py)
    
    num_encoder_tokens = size_voc_mrs # Number of unique input tokens
    num_decoder_tokens = size_voc_ref # Number of unique output tokens
    
    max_encoder_seq_length = max(len(txt) for txt in x_train_seq) # Max Sequence length for inputs
    max_decoder_seq_length = max(len(txt) for txt in y_train_seq) # Max Sequence length for outputs
    
    
    
    print('Start encoding')
    x_train_to_encode = x_train_seq + y_train_seq #+ x_test_seq
    
    create_embeddings(x_train_to_encode,
                                            path_to_embeddings,
                                            path_to_vocab,
                                            path_to_embedding_model,
                                            size=100,
                                            min_count=2,
                                            window=5,
                                            iter=1)



    # Load embedding
    print('load embedding')
    embedding_model = load_embedding_model(path_to_embedding_model)
    weights = embedding_model.syn0 
    
    # Load dicts
    
    padding_vec = np.zeros(embedding_model.syn0.shape[1])  
    
    print('encode X_train + padding')
    X_train=[]
    for mrs in x_train_seq:
        new_mr=[]
        for word in mrs:
            new_mr.append(embedding_model[word])
        padded_new_mr = add_padding(new_mr, padding_vec, max_encoder_seq_length)
        X_train.append(padded_new_mr)
    X_train = np.array(X_train)
    
    
    print('encode Y_train')

    X_train_ohe = np.zeros((len(x_train_seq), max_encoder_seq_length, num_encoder_tokens), dtype=np.int8)
    Y_train = np.zeros((len(y_train_seq), max_decoder_seq_length, num_decoder_tokens), dtype=np.int8)
    for i, (x, y) in enumerate(zip(x_train_seq, y_train_seq)):
        for t, word in enumerate(x):
            X_train_ohe[i, t, input_token_index[word]] = 1
        for t, word in enumerate(y):
            #print(t)
            #print(y)
            Y_train[i, t, target_token_index[word]] = 1


            
            
    # =============================================================================
    #      LSTM model     
    # =============================================================================

    print('Build LSTM model')
    max_input_seq_len = max_encoder_seq_length 
    max_output_seq_len = max_decoder_seq_length
    hidden_layer_size = 300
    
    wgt = weights.shape[1]
    input = Input(shape=(max_input_seq_len, num_encoder_tokens))
    
    # -- ENCODER --
    
    print('Set encoder')
    encoder = Bidirectional(LSTM(units=hidden_layer_size,
                                 dropout=0.2,
                                 recurrent_dropout=0.2,
                                 return_sequences=True),
                            merge_mode='concat')(input)
        
    flattened = Flatten()(encoder)
    
    
    attention = []
    for i in range(max_output_seq_len):
        weighted = Dense(max_input_seq_len, activation='softmax')(flattened)
        unfolded = Permute([2, 1])(RepeatVector(hidden_layer_size * 2)(weighted))
        multiplied = multiply([encoder, unfolded])
        summed = Lambda(lambda x: K.sum(x, axis=-2))(multiplied)
        attention.append(Reshape((1, hidden_layer_size * 2))(summed))
    
    attention_out = concatenate(attention, axis=-2)
    
    # -- DECODER --
    
    print('Set decoder')
    decoder = LSTM(units=hidden_layer_size,
                   dropout=0.2,
                   recurrent_dropout=0.2,
                   return_sequences=True)(attention_out)
    
    decoder = Dense(num_decoder_tokens,
                    activation='softmax')(decoder)
    
    model = Model(inputs=input, outputs=decoder)
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    
    model.summary()
    
    
    
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    filepath = 'trained_model.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    
    
    # ---- TRAIN ----
    print('Start fitting trainset...')
    model.fit(X_train_ohe, Y_train,
              batch_size=128,
              epochs=5,
              callbacks=callbacks_list)
    
    model_json = model.to_json()
    with open(os.path.join(path_to_model,"model.json"),"w") as json_file:
        json_file.write(model_json)
        
    model.save_weights(os.path.join(path_to_model,"model.h5"))

            
    