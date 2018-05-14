Assignment 2: ASBA System

>> pdf version available in 'resources' folder


Authors: Inès KALLALA, Iza OUAHÈS, Clotilde PETY, Chuyi XU


This README document contains 4 sections:
1.	Description of the system
2.	Final model
3.	How to run classifier.py?


1.	Description of the system: 

i.	First, we tried a model where we cleaned the sentences contained in the train and in the test samples using the remove_characters(), sorted_stopwords() and clean() functions. We then applied a TF-IDF Vectorizer on these cleaned sentences to boost the weights of the frequent words that appear in the entire vocabulary. To take into account the aspect associated to each review, we also decided to transform target words into dummy variables and therefore to consider them as features along with the weights associated for each word after applying a TF-IDF Vectorizer. We used a MultinomialNB() classifier and obtained an accuracy of 77.12% on the test set.

ii.	However, we thought that the 1st approach misled our model. Let’s take an example with this sentence: “The food was good but the service sucks!”
The above-mentioned sentence contains 2 different polarities: 
-	Regarding the target word “food”, the polarity is positive
-	However, regarding the target word “service”, the polarity is negative
In this respect, training a model on full sentences is not really accurate and relevant because when training and testing, it will allocate same weights for words that actually do not refer to the same target word.
As a consequence, we thought that it would be interesting to study the dependencies of the target word (and thus the aspect). Using the above-mentioned example “The food was good but the service sucks!”, it would give:
-	Food: was, good
-	Service: sucks
And to keep, for each target word, only its dependent words. To do that, we used 2 different parsers:
-	StanfordDependencyParser
-	spaCy parser
You can see the results of the 2 parsers in the saved_train.csv file. The “parsed_sentence” column contains the dependencies spotted by the Stanford parser while the “Spacy_parsed_list” contains the dependencies spotted by the spaCy parser. It turns out that the StanfordDependencyParser outperformed the spaCy parser to detect the dependencies of target words. Note that the code for the spaCy parser is contained in classifier.py (Choice 3, line 131). Its output is contained in the column “Spacy_parsed_list”. 
 


Then, we keep only the words that depend on the target word. These words are used as features. Below is a summary table of the results obtained on the test set:


Training set output	: Stanford parser	| Stanford parser |	Cleaned sentences
Test set output	    : Cleaned sentences	| Stanford parser |	Cleaned sentences
Accuracy	        : 78.72%	        | 76.06%	      | 77.12%

In an attempt to improve our results, we also tried a voting classifier with:    
•	clf1 = LogisticRegression(random_state=1)
•	clf2 = RandomForestClassifier(random_state=1)
•	clf3 = MultinomialNB()
•	clf4= SVC()
ACCURACY: 0.7845
Exec time: 9.27 s.
However, the result turns out to be less accurate. 
	
2.	Final model: To sum up we built the final model in 4 steps:
-	We pre-preprocessed the training set with the remove_characters(), the remove_characters_no_low() that remove figures and punctuation from the sentences and that lower the cases. 
-	We applied the Stanford parser on the training set. The output of the parser is the dependencies. We then extract the words that are linked to the target words. We transformed these dependent words with a CountVectorizer so that they become the new features of the training set. We fit the classifier (LogisticRegression()) on the transformed training set.
-	Regarding the test set, we clean the sentences using the remove_characters(), the remove_characters_no_low() and the clean() functions. Note that the clean() function removes a customized list of stopwords (you can see the list in the code). Note that we cannot apply this function on the training set because it would break the dependencies.
-	Finally, we predict on the test set and assess the test accuracy.
-	
3.	How to run classifier.py? 
Python script (2 steps)	

classifier.py : input the path to saved_train.csv in the variable “path_to_pre_parsed” (line 1)	
	The csv will be loaded when we call the training set in the train() function. 
	We decided to use this process because applying the Stanford parser on raw data on our laptop takes at least 45 minutes. 
	All the code needed to transform the training set is displayed as comments (see “Choice 2”, line 104).
	
tester.py : Update “datadir” (line 23)	


