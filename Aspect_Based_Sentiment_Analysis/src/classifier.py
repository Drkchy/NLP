path_to_pre_parsed=r'C:\Users\Chuyi\Documents\DSBA\T2\NLP\Exercise_2\saved_train.csv'

import pandas as pd
import numpy as np
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import TransformerMixin 
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV

import random
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('wordnet')

# Download and Load Stanford dependency parser if you want to compute dependencies line 107-127
#from nltk.parse.stanford import StanfordDependencyParser
#path_to_jar = 'C:\ProgramData\Anaconda3\stanford-parser-full-2018-02-27\stanford-parser.jar'
#path_to_models_jar = 'C:\ProgramData\Anaconda3\stanford-parser-full-2018-02-27\stanford-parser-3.9.1-models.jar'
#dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)

import spacy
from spacy.pipeline import Tagger
from spacy.tokenizer import Tokenizer
import en_core_web_sm
nlp = en_core_web_sm.load()
tagger = Tagger(nlp.vocab)
tokenizer = Tokenizer(nlp.vocab)


def remove_characters(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(\s?[0-9]+\.?[0-9]*)", " ", tweet).lower().split())

def remove_characters_no_low(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(\s?[0-9]+\.?[0-9]*)", " ", tweet).split())

def sorted_stopwords(stop_words):
    #Since negative words are useful to predict the label 'negative', we remove them from stopwords
    negative_words = ["needn't","aren't", "ain", "not", "shouldn", "mightn", "weren","wouldn't","don't", "hasn't", 'wasn','couldn', "won't", "hadn't", "shan't", "doesn't","haven't","wasn't", 'but', 'against','isn', 'didn', 'no', "mustn't", "isn't", "mightn't", "mustn", "couldn't",'don','nor', 'haven', 'doesn','hasn',  "didn't", 'aren', 'wouldn', 'needn','hadn', "shouldn't", "weren't"]
    stopwords_clean = []
    for word in stop_words:
        if word not in negative_words:
            stopwords_clean.append(word)
    return stopwords_clean

def clean(sentence):
    clean_stopwords=sorted_stopwords(stop_words)
    
    lmtzr = WordNetLemmatizer()
    new_sentence =" ".join([lmtzr.lemmatize(word) for word in sentence.split()])
    #tokenize
    word_tokens = word_tokenize(new_sentence)
    wordlist = [word for word in word_tokens if word not in clean_stopwords]
    return wordlist
    
def log(x):
    #can be used to write to log file
    print(x)


class Classifier:
    """The Classifier"""

    #############################################
    #aspects='AMBIENCE#GENERAL','DRINKS#PRICES','DRINKS#QUALITY','DRINKS#STYLE_OPTIONS','FOOD#PRICES','FOOD#QUALITY','FOOD#STYLE_OPTIONS','LOCATION#GENERAL','RESTAURANT#GENERAL','RESTAURANT#MISCELLANEOUS','RESTAURANT#PRICES','SERVICE#GENERAL'
    
    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""
        #load the two files        
        cols = ['polarity','category','term_target','start_end','sentence']
        train_data=pd.read_csv(trainfile, sep='\t', header=None, names=cols)
        
#        train_data['cleaned_sentence']="" 
#        train_data['parsed_sentence']="" #Stanford Sparser tuples
#        train_data['parsed_sentence_list']="" #List of dependent words from the sparsing
#        train_data['Spacy_parsed_list']="" #Spacy parser dependent words from the sparsing
#        
#        Rename columns
#        train_data.columns = ['polarity','category','term_target','start_end','sentence','cleaned_sentence','parsed_sentence','parsed_sentence_list','Spacy_parsed_list']
        
        #===========================================================
        #Choice 1: CLASSICAL features creation
        #Remove: punctuations, characters, numbers, uppercase
    
#        for i in range (0, len(train_data)):
#            result=remove_characters(train_data.iloc[i]['sentence'])
#            cleaned_sentence= clean(result)
#            train_data.iloc[i,5]=' '.join(cleaned_sentence)
        
        #===========================================================
        #Choice 2: Stanford parser on raw sentences
        #https://cs224d.stanford.edu/reports/WangBo.pdf
        
#        Fill train_data['parsed_sentence']
#        for i in range (0, len(train_data)):
#            result=remove_characters_no_low(train_data.iloc[i]['sentence'])
#            #result=train_data.iloc[i]['sentence']
#            parse = dependency_parser.raw_parse(result)
#            dep = parse.__next__()            
#            train_data.iloc[i,6]=list(dep.triples()) 
#        
#        Fill train_data['parsed_sentence_list']
#        for i in range (0, len(train_data)):
#            target_list=[w for w in word_tokenize(train_data.iloc[i]['term_target'])]
#            tuples_list=train_data.iloc[i]['parsed_sentence']
#            new_list=[]
#            for w in target_list:
#                new_list.append(w)
#                for a,b,c in tuples_list:
#                    if a[0]==w:
#                        new_list.append(c[0])
#                    if c[0]==w:
#                        new_list.append(a[0])
#            train_data.iloc[i,7]=' '.join([i.lower() for i in new_list])
        
    
        #===========================================================
#        Choice 3: Spacy parser on raw sentences
#        Fill train_data['Spacy_parsed_list']
#        for i in range (0, len(train_data)):
#            target_list=[w for w in word_tokenize(train_data.iloc[i]['term_target'])]
#            result=remove_characters_no_low(train_data.iloc[i]['sentence'])
#            new_list=[]
#            parsedEx = nlp(result)
#            for token in parsedEx:
#                if token.orth_ in target_list:
#                    new_list.append(tuple([token.orth_]+[token.head.orth_]+[t.orth_ for t in token.lefts]+[t.orth_ for t in token.rights]))
#            train_data.iloc[i,8]=' '.join([i.lower() for sub in new_list for i in sub])
        
        
        #===========================================================
        #=============== LOAD TRAIN DATA (Parsing done)=============
        #===========================================================
        
        #train_data.to_csv(r'C:\Users\Chuyi\Documents\DSBA\T2\NLP\Exercise_2\saved_train.csv',sep=';',index_label=False,encoding='utf-8')
        
        loaded_data=pd.read_csv(path_to_pre_parsed,sep=';')
        
        train_data=loaded_data


        #===========================================================
        #====================== MODEL SETTING ======================
        #===========================================================
        
        #Choice 1 : Train the model on all cleaned sentences 
        
#        # Initialize a TfidfVectorizer object: tfidf_vectorizer
#        tfidf_vectorizer = TfidfVectorizer(stop_words="english",max_df=0.7)
#        # Transform the training data: tfidf_train 
#        tfidf_train = tfidf_vectorizer.fit_transform(train_data["cleaned_sentence"])
#        y_train=train_data["polarity"]
#        tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())
#        
#        # Creation of dummies features that take into account the aspect of the sentence 
#        dummies = pd.get_dummies(train_data["category"])
#        X_train = dummies.join(tfidf_df)
#        
#        # Fit the train-set columns (X_train) to the labels (y_train)
#        MultinomialNB = MultinomialNB()
#        MultinomialNB.fit(X_train, y_train)

        
        
        #Choice 2 or 3 : Train the model on all dependent words according to term_target 

        # BOW is used to create features according to the words linked to term_target
        self.bow_transformer = CountVectorizer()
        
        # Transform the training data into bow_train 
        bow_train = self.bow_transformer.fit_transform(train_data["parsed_sentence_list"]) # Stanford parser
        #bow_train = self.bow_transformer.fit_transform(train_data["Spacy_parsed_list"]) # Spacy parser

        #Discretize the labels with a mapping
        #mapping = {"positive": 1, "neutral": 0, "negative": -1}
        y_train=train_data['polarity']
        #y_train=train_data['polarity'].map(mapping)
        
        bow_df = pd.DataFrame(bow_train.A, columns=self.bow_transformer.get_feature_names())
        
        # Keep the dummies features that take into account the aspect of the sentence 
        dummies = pd.get_dummies(train_data["category"])
        X_train = dummies.join(bow_df)
        
        self.nb_classifier = LogisticRegression()
        #self.nb_classifier = RandomForestClassifier()
        self.nb_classifier.fit(X_train, y_train)
        

    
    def predict(self, devfile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        cols = ['polarity','category','term_target','start_end','sentence']
        dev_data=pd.read_csv(devfile, sep='\t', header=None, names=cols)
        dev_data['cleaned_sentence']=""
        dev_data['parsed_sentence']="" #Stanford Sparser tuples
        dev_data['parsed_sentence_list']="" #List of dependent words from the sparsing
        dev_data['Spacy_parsed_list']="" #Spacy parser dependent words from the sparsing
        
        #Rename columns
        #train_data.columns = ['polarity','category','term_target','start_end','sentence','cleaned_sentence','parsed_sentence','parsed_sentence_list','Spacy_parsed_list']
        
        #Choice 1: CLASSICAL features creation
        #Remove: punctuations, characters, numbers, uppercase
        for i in range (0, len(dev_data)):
            result=remove_characters(dev_data.iloc[i]['sentence'])
            cleaned_sentence= clean(result)
            dev_data.iloc[i,5]=' '.join(cleaned_sentence)

#        tfidf_test = tfidf_vectorizer.transform(dev_data["cleaned_sentence"])
#        tfidf_df_test = pd.DataFrame(tfidf_test.A, columns=tfidf_vectorizer.get_feature_names())
#        
#        dummies_test = pd.get_dummies(dev_data["category"])
#        X_test = dummies_test.join(tfidf_df_test)
#        y_test = dev_data["polarity"]
#        
#        pred = MultinomialNB.predict(X_test)
#        score = metrics.accuracy_score(y_test, pred)
#        #precision, recall, accuracy, f1 = test_classifier(X_train, y_train, X_test, y_test, MultinomialNB())
#        print(score)



        #Choice 2: Stanford sparser on raw sentences
        #Fill : dev_data[parsed_sentence]

#        for i in range (0, len(dev_data)):
#            result=remove_characters_no_low(dev_data.iloc[i]['sentence'])
#            #result= dev_data.iloc[i]['sentence']
#            parse = dependency_parser.raw_parse(result)
#            dep = parse.__next__()            
#            dev_data.iloc[i,6]=list(dep.triples()) 
#
#        #Fill : dev_data[parsed_sentence_list]
#        for i in range (0, len(dev_data)):
#            target_list=[w for w in word_tokenize(dev_data.iloc[i]['term_target'])]
#            tuples_list=dev_data.iloc[i]['parsed_sentence']
#            new_list=[]
#            for w in target_list:
#                new_list.append(w)
#                for a,b,c in tuples_list:
#                    if a[0]==w:
#                        new_list.append(c[0])
#                    if c[0]==w:
#                        new_list.append(a[0])
#            dev_data.iloc[i,7]=' '.join([i.lower() for i in new_list])
#
#
#
#        #Choice 3: Spacy sparser on raw sentences.
#        #Fill : dev_data[Spacy_parsed_list]
#        for i in range (0, len(dev_data)):
#            target_list=[w for w in word_tokenize(dev_data.iloc[i]['term_target'])]
#            result=dev_data.iloc[i]['sentence']
#            new_list=[]
#            parsedEx = nlp(result)
#            for token in parsedEx:
#                if token.orth_ in target_list:
#                    new_list.append(tuple([token.orth_]+[token.head.orth_]+[t.orth_ for t in token.lefts]+[t.orth_ for t in token.rights]))
#            dev_data.iloc[i,8]=' '.join([i.lower() for sub in new_list for i in sub])
        
        
        
        
        bow_test = self.bow_transformer.transform(dev_data["cleaned_sentence"])
        bow_df_test = pd.DataFrame(bow_test.A, columns=self.bow_transformer.get_feature_names())

        dummies_test = pd.get_dummies(dev_data["category"])
        X_test = dummies_test.join(bow_df_test)
        
        #mapping = {"positive": 1, "neutral": 0, "negative": -1}
        y_test=dev_data["polarity"]
        #y_test = dev_data["polarity"].map(mapping)
        
        pred = self.nb_classifier.predict(X_test)
        score = metrics.accuracy_score(y_test, pred)
        

        #à rajouter
        return pred


