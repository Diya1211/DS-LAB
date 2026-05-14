import pandas as pd

import numpy as np

import nltk

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer,WordNetLemmatizer

from nltk import pos_tag

from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download ('punkt')

nltk.download ('averaged_perceptron_tagger')

nltk.download ('stopwords')

nltk.download ('wordnet')

document = """
Natural language processing (NLP) is a subfield of artificial intelligence (AI)
that focuses on the interaction between computers and humans using natural language.
It involves the analysis, understanding, and generation of human language,
enabling machines to process and comprehend text in a meaningful way.
NLP techniques are widely used in various applications such as sentiment analysis,
machine translation, chatbots, and information retrieval.
Preprocessing is an essential step in NLP, which involves tokenization,
part-of-speech tagging, stop words removal, stemming, and lemmatization.
"""

tokens = word_tokenize (document)

pos_tags = pos_tag (tokens)

stop_words = set (stopwords.words ('english'))

filtered_tokens = [

token for token in tokens

if token.lower ()not in stop_words

]

stemmer = PorterStemmer ()

stemmed_tokens = [

stemmer.stem (token)

for token in filtered_tokens

]

lemmatizer = WordNetLemmatizer ()

lemmatized_tokens = [

lemmatizer.lemmatize (token)

for token in filtered_tokens

]

print ("Original Document:\n")

print (document)

print ("\nTokens:\n")

print (tokens)

print ("\nPOS Tags:\n")

print (pos_tags)

print ("\nFiltered Tokens:\n")

print (filtered_tokens)

print ("\nStemmed Tokens:\n")

print (stemmed_tokens)

print ("\nLemmatized Tokens:\n")

print (lemmatized_tokens)

documents = [

"Natural language processing NLP is a subfield of artificial intelligence.",

"It focuses on the interaction between computers and humans using natural language.",

"NLP techniques are widely used in various applications such as sentiment analysis and machine translation.",

"Preprocessing is an essential step in NLP."

]

vectorizer = TfidfVectorizer ()

tfidf_matrix = vectorizer.fit_transform (documents)

feature_names = vectorizer.get_feature_names_out ()

for i,doc in enumerate (documents):

    print (f"\nDocument {i +1 }:")

    for j,term in enumerate (feature_names):

        tfidf_value = tfidf_matrix[i,j ]

        if tfidf_value >0 :

            print (f"{term }: {tfidf_value }")
