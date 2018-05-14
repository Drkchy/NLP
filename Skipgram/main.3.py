
from __future__ import division


import argparse

import pandas as pd
import pickle

import numpy as np
import timeit

# define our method
def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text

def remove_punctuation(text,punct):
    my_str = text
    no_punct = ""
    for char in my_str:
        if char not in punct:
            no_punct = no_punct + char    
    return no_punct

# 1. Load text
def text2sentences(path):
    # We assume that the training text is written in English
    # Which means, for example, that we won't take into account French abbreviations s.t. M., Mme, etc
    dico = {'“':' ', '”':' ', 'Mrs.' : 'Mrs', 'Mr.' : 'Mr', 'Ms.': 'Ms', 'Sir.': 'Sir', 'Dr.': 'Dr', 'Jr.' : 'Jr', 'Sr.': 'Sr'}
    punctuations_1 = '''!()-[]{};:'"\,<>/?@#$%^&*_~'''
    punctuations_2 = '.'
    stopwords = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'}

    
    corpus_to_string = []
    with open(path) as f:
        corpus = f.read()
        corpus_without_punctuation = remove_punctuation(corpus,punctuations_1)
        corpus_without_punctuation = corpus_without_punctuation.replace('\n',' ')
        corpus_clean = replace_all(corpus_without_punctuation, dico)

        # Find the sentences
        while corpus_clean.find('.') != -1:
            index = corpus_clean.find('.')
            corpus_to_string.append(corpus_clean[:index+1])
            corpus_clean = corpus_clean[index+1:]
       
    sentences = []
    for sentence in corpus_to_string:
        sentence=remove_punctuation(sentence,punctuations_2)
        sentences.append([word for word in sentence.lower().strip().split() if word not in stopwords])
        #print(type(sentences))

    #Finally: Remove stopwords 
    #for i in range(len(sentences)):    
    #   sentences[i]=list(lambda x: x not in stopwords, sentences[i])
    return sentences

def dictionary_words(sentences):
    dictionary_words ={}
    for word in np.concatenate(sentences, axis=0):
        dictionary_words[word] = dictionary_words.get(word,0)+1
    
    return dictionary_words

def loadPairs(path):
    data = pd.read_csv(path,delimiter='\t')
    pairs = zip(data['word1'],data['word2'],data['similarity'])
    return pairs

class mySkipGram:
    def __init__(self, sentences, nEmbed=100, negativeRate=5,winSize=5):
        
        self.nEmbed = nEmbed
        self.negativeRate = negativeRate
        self.winSize = winSize
        self.context = 2*winSize
        self.sentences = sentences
        
        self.first_dict=dictionary_words(self.sentences)
        
        # dictionary_words(sentences) contains the count for each key
        self.new_dictionary = {k: v for k, v in self.first_dict.items()}
              
        self.vocabulary = list(self.new_dictionary)
        self.V=len(self.vocabulary)
        print(self.V)

        # Now, we want to retrieve an index for each word of the vocabulary, in the ORDER of the vocabulary
        self.dictionary_word_indices = {value: key for key, value in enumerate(self.vocabulary)}
         
       
        #For each target word of the vocabulary, look up for its context words (careful: it has to take into account that the target word and its context words must be in the same sentence)
        self.context=[]
        for sentence in self.sentences:
            for index_1, theWord  in enumerate(sentence): # first sentence  generate a word and its context
                
                self.context.append((theWord, 
                                     [context_word for index_2, 
                                      context_word in enumerate(sentence) 
                                      if (np.abs(index_1-index_2)<=self.winSize) 
                                      and index_1 != index_2]))

        #print("sentence{}, theWord, context{}".format(sentence,theWord, self.context))
        print(len(self.context))
        # initializing weights
        self.inputhidden_weights = np.random.uniform(size = (self.V, self.nEmbed))
        self.outputhidden_weights = np.random.uniform(size = (self.nEmbed, self.V))

    def get_index(self, context_word):
        context_word_index = []
        for i in range(len(context_word)):
            context_word_index.append(self.dictionary_word_indices.get(context_word[i]))
        return context_word_index

    def negative_sampling(self,context_word):
         # Return the list of the probabilities that will allow us to pick negative samples        
        #print("compute the probabilities")
                
        dico= [i**0.75 for i in self.new_dictionary.values()]
        self.proba_list=[]
        for v in self.new_dictionary.values():
            prob=np.power(v, 0.75)/(sum(dico))
            self.proba_list.append(prob)                
        self.negative_sample= np.random.choice(self.V, size = self.negativeRate, p=self.proba_list)              
        return [(self.dictionary_word_indices[context_word], 1)] + [(negative_word, 0) for negative_word in self.negative_sample]

    def sigmoid(self, x, y):
        return 1/(1+np.exp(-np.sum(x*y.T)))  

    def train(self,stepsize=0.01, epochs=5):
    # We used the paper "word2vec Parameter Learning" by Xin Rong to know how to perform backpropagation in a skip-gram model   
        
    # Measure the time for training
        start = timeit.default_timer()
        
        for epoch in range(epochs):
            
            # Scan the 6,000 contexts
            for theWord, context_words in self.context:
                
                h = self.inputhidden_weights[self.dictionary_word_indices[theWord],:]
                
                for context_word in context_words:
                    
                    # Pick negative samples
                    training_samples = self.negative_sampling(context_word)
                    
                    # First, compute the erreur of the hidden-output layer weights
                    # Use eq(61) in Xin Rong to compute the error
                    error = np.sum([(self.sigmoid(self.outputhidden_weights[:,j], h) - neg_j)*self.outputhidden_weights[:,j] for j, neg_j in training_samples], axis = 0)
                    
                    # Then, update outputhidden_weights (cf equation (59) in Xin Rong)
                    for j, neg_j in training_samples:
                        
                        self.outputhidden_weights[:,j] = self.outputhidden_weights[:,j] - stepsize * (self.sigmoid(self.outputhidden_weights[:,j], h)-neg_j) * h.T
                    
                    # Finally update the hidden-input layer weights by backpropagating the error
                    # Use eq(35) i nXin Rong to backpropagate the error to the input-hidden layer weights
                    self.inputhidden_weights[j, :] = self.inputhidden_weights[j, :] - stepsize * error.T    
                
        stop= timeit.default_timer()
        print('running time',stop - start, 's')

    def save(self,path):
        pickle.dump(self,open(path, 'wb'))
  
    def similarity(self,word1,word2):
        """
        computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
        # use cos distance to compute similarity
        vect1= self.inputhidden_weights[self.dictionary_word_indices[word1],:]
        vect2= self.inputhidden_weights[self.dictionary_word_indices[word2],:]
        cos_distance = np.dot(vect1, vect2.T)/ float(np.linalg.norm(vect1)*np.linalg.norm(vect2))
        return (1+cos_distance)/2

    @staticmethod
    def load(path):
        f = open(path,'rb')
        obj = pickle.load(f)
        f.close()
        return obj



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences(opts.text)
        sg = mySkipGram(sentences)
        sg.train()
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)

        sg = mySkipGram.load(opts.model)
        
        for a,b,_ in pairs:
            print(sg.similarity(a,b))