# -*- coding: utf-8 -*-
"""
Created on Tue May 16 20:17:47 2017

@author: Charles
"""

from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize("is", 'v')

lemmatizer.lemmatize('provided', 'v')

lemmatizer.lemmatize('using', 'v')

lemmatizer.lemmatize('a', 'v')

lemmatizer.lemmatize('contracting', 'v')