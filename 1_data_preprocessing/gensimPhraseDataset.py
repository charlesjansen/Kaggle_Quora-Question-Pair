# -*- coding: utf-8 -*-

#https://radimrehurek.com/gensim/models/phrases.html

import numpy as np
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from pylab import plot, show, subplot, specgram, imshow, savefig
import re
from gensim.models import Phrases

#importing data with spacy lemmatizer done
print("load train")
train = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/train_spacyLemma.csv', header=0) 
print("load test")
test = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/test_spacyLemma.csv', header=0) 

#cleaning
def regex(text):
    text = re.sub("？", "?", text) 
    text = re.sub("\(.*?\)", " ", text) 
    text = re.sub("…", "", text) 
    text = re.sub("’", "'", text) # special single quote
    text = re.sub("`", "'", text) # special single quote
    text = re.sub("“", '"', text) # special double quote
    text = re.sub("é", "e", text) 
    text = re.sub("è", "e", text) 
    # standard
    text = re.sub(r"(\b)([Aa]re|[Cc]ould|[Dd]id|[Dd]oes|[Dd]o|[Hh]ad|[Hh]as|[Hh]ave|[Ii]s|[Mm]ight|[Mm]ust|[Ss]hould|[Ww]ere|[Ww]ould)n't", r"\1\2 not", text)
    text = re.sub(r"(\b)([Hh]e|[Ii]|[Ss]he|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'ll", r"\1\2 will", text)
    text = re.sub(r"(\b)([Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'re", r"\1\2 are", text)
    text = re.sub(r"(\b)([Ii]|[Ss]hould|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Ww]ould|[Yy]ou)'ve", r"\1\2 have", text)
    # non-standard
    text = re.sub(r"(\b)([Cc]a)n't", r"\1\2n not", text)
    text = re.sub(r"(\b)([Ii])'m", r"\1\2 be", text)
    text = re.sub(r"(\b)([Ll]et)'s", r"\1\2 us", text)
    text = re.sub(r"(\b)([Ww])on't", r"\1\2ill not", text)
    text = re.sub(r"(\b)([Ss])han't", r"\1\2hall not", text)
    text = re.sub(r"(\b)([Yy])(?:'all|a'll)", r"\1\2ou all", text)
    text = re.sub(r" quikly ", " quickly ", text)
    text = re.sub(r" unseccessful ", " unsuccessful ", text)
    text = re.sub(r" addmision ", " admission ", text)
    text = re.sub(r" insititute ", " institute ", text)
    text = re.sub(r" connectionn ", " connection ", text)
    text = re.sub(r" permantley ", " permanently ", text)
    text = re.sub(r" sylabus ", " syllabus ", text)
    text = re.sub(r" sequrity ", " security ", text)
    text = re.sub(r" latop", " laptop", text)
    text = re.sub(r" programmning ", " programming ", text)  
    text = re.sub(r" begineer ", " beginner ", text)  
    text = re.sub(r" qoura ", " Quora ", text)
    text = re.sub(r" wtiter ", " writer ", text)  
    text = re.sub(r" litrate ", " literate ", text)  
    text = re.sub("e-mail", " email ", text, flags=re.IGNORECASE)
    text = re.sub("(the[\s]+|The[\s]+)?U\.S\.A\.", " america ", text, flags=re.IGNORECASE)
    text = re.sub(" usa ", " america ", text)
    text = re.sub("(the[\s]+|The[\s]+)?United State(s)?", " america ", text, flags=re.IGNORECASE)
    text = re.sub("\(s\)", " ", text, flags=re.IGNORECASE)
    text = re.sub("[c-fC-F]\:\/", " disk ", text)
    text = re.sub("\'ll", " be ", text)
    text = re.sub("e\.g\.", " eg ", text, flags=re.IGNORECASE)
    text = re.sub("b\.g\.", " bg ", text, flags=re.IGNORECASE)
    text = re.sub("e \.g \.", " eg ", text, flags=re.IGNORECASE)
    text = re.sub("b \.g \.", " bg ", text, flags=re.IGNORECASE)
    text = re.sub("e \. g \.", " eg ", text, flags=re.IGNORECASE)
    text = re.sub("b \. g \.", " bg ", text, flags=re.IGNORECASE)
    text = re.sub("([0-9]*?)\.[0-9]*", r"\1", text)
    text = re.sub(r"(\W|^)([0-9]+)[kK](\W|$)", r"\1\g<2>000\3", text) # better regex provided by @armamut
    text = re.sub("\'s", " ", text) # we have cases like "Sam is" or "Sam's" (i.e. his) these two cases aren't separable, I choose to compromise are kill "'s" directly
    text = re.sub(" whats ", " what be ", text, flags=re.IGNORECASE)
    text = re.sub("\'ve", " have ", text)
    text = re.sub("can't", "can not", text)
    text = re.sub("n't", " not ", text)
    text = re.sub("i'm", "-PRON- be", text, flags=re.IGNORECASE)
    text = re.sub("\'re", " be ", text)
    text = re.sub("\'d", " would ", text)
    text = re.sub("\'ll", " will ", text)
    text = re.sub('(?<=[0-9])\,(?=[0-9])', "", text)
    text = re.sub('\$', " dollar ", text)
    text = re.sub('\%', " percent ", text)
    text = re.sub('\&', " and ", text)
    text = re.sub(r" (the[\s]+|The[\s]+)?US(A)? ", " america ", text)
    text = re.sub(r" uk ", " england ", text, flags=re.IGNORECASE)
    text = re.sub(r" imrovement ", " improvement ", text, flags=re.IGNORECASE)
    text = re.sub(r" intially ", " initially ", text, flags=re.IGNORECASE)
    text = re.sub(r" dms ", " direct message ", text, flags=re.IGNORECASE)  
    text = re.sub(r" demonitization ", " demonetization ", text, flags=re.IGNORECASE) 
    text = re.sub(r" actived ", " active ", text, flags=re.IGNORECASE)
    text = re.sub(r" kms ", " kilometer ", text, flags=re.IGNORECASE)
    text = re.sub(r" cs ", " computer science ", text, flags=re.IGNORECASE) 
    text = re.sub(r" upvote", " up vote", text, flags=re.IGNORECASE)
    text = re.sub(r" iPhone ", " phone ", text, flags=re.IGNORECASE)
    text = re.sub(r" \0rs ", " rs ", text, flags=re.IGNORECASE)
    text = re.sub(r" calender ", " calendar ", text, flags=re.IGNORECASE)
    text = re.sub(r" ios ", " operating system ", text, flags=re.IGNORECASE)
    text = re.sub(r" programing ", " programming ", text, flags=re.IGNORECASE)
    text = re.sub(r" bestfriend ", " best friend ", text, flags=re.IGNORECASE)
    text = re.sub(r" III ", " 3 ", text)
    text = re.sub(r" iii ", " 3 ", text)
    text = re.sub(r" II ", " 2 ", text)
    text = re.sub(r" ii ", " 2 ", text)
    text = re.sub(r" j k ", " jk ", text, flags=re.IGNORECASE)
    text = re.sub(r" j\.k\. ", " jk ", text, flags=re.IGNORECASE)
    text = re.sub(r" quikly ", " quickly ", text)
    text = re.sub(r" unseccessful ", " unsuccessful ", text)
    text = re.sub(r" demoniti[\S]+ ", " demonetization ", text, flags=re.IGNORECASE)
    text = re.sub(r" demoneti[\S]+ ", " demonetization ", text, flags=re.IGNORECASE)  
    text = re.sub(r" addmision ", " admission ", text)
    text = re.sub(r" insititute ", " institute ", text)
    text = re.sub(r" connectionn ", " connection ", text)
    text = re.sub(r" permantley ", " permanently ", text)
    text = re.sub(r" sylabus ", " syllabus ", text)
    text = re.sub(r" sequrity ", " security ", text)
    text = re.sub(r" undergraduation ", " undergraduate ", text) # not typo, but GloVe can't find it
    text = re.sub(r"(?=[a-zA-Z])ig ", "ing ", text)
    text = re.sub(r" latop", " laptop", text)
    text = re.sub(r" programmning ", " programming ", text)  
    text = re.sub(r" begineer ", " beginner ", text)  
    text = re.sub(r" qoura ", " Quora ", text)
    text = re.sub(r" wtiter ", " writer ", text)  
    text = re.sub(r" litrate ", " literate ", text) 
    text = re.sub(r" upsc ", " civil service ", text) 
    text = re.sub("-", " ", text)
    text = re.sub("[ ]{1,}", " ", text)
    text = re.sub(" ' ", " ", text)
    text = re.sub(" \" ", " ", text)
    text = re.sub(" ? ", " ", text)
    text = re.sub(" \/ ", " ", text)
    text = re.sub(" : ", " ", text)
    text = re.sub(" @ ", " at ", text)
    text = re.sub(" before PRON die ", " must do ", text)
    text = re.sub(" dark knight ", " batman ", text)
    text = re.sub(" about ", " ", text)
    return(text)


#text= "what would a trump presidency mean for current international master ’s student on an f1 visa ? "
#print(regex(text))


print("regex on train question 1")
train['question1'] = train.question1.apply(lambda x: regex(x))
print("regex on train question 2")
train['question2'] = train.question2.apply(lambda x: regex(x))

print("regex on test question 1")
test['question1'] = test.question1.apply(lambda x: regex(x))
print("regex on test question 2")
test['question2'] = test.question2.apply(lambda x: regex(x))

#==============================================================================
# 
# 
# #to do, remove stop word just for creating the model (not in the questions when processing after)
# 
# questions = pd.concat([train.question1, train.question2, test.question1, test.question2]).apply(str).fillna("empty").tolist()
# sentence_stream_q = [question.split(" ") for question in questions]
# 
# #creating the bigrame model
# print("creating bigram model")
# bigram_model = Phrases(sentence_stream_q)
# bigram_model.save("F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/bigram/spacy_bigram.gensim")
# 
# bigram_model = Phrases.load("F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/bigram/spacy_bigram.gensim")
# 
# 
# 
# #creating the tigrame model
# print("creating trigram model")
# trigram_model = Phrases(bigram_model[sentence_stream_q])
# trigram_model.save("F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/bigram/spacy_trigram.gensim")
# 
# trigram_model = Phrases.load("F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/bigram/spacy_trigram.gensim")
# 
# 
# 
# def do_bigram(text):
#     return(" ".join(bigram_model[text.split(" ")]))
# 
# def do_trigram(text):
#     return(" ".join(trigram_model[bigram_model[text.split(" ")]]))
# 
# #applying
# print("trigram_Q1")
# train["trigram_Q1"] =  train.apply(lambda x: do_trigram(x['question1']), axis=1)
# print("trigram_Q2")
# train["trigram_Q2"] =  train.apply(lambda x: do_trigram(x['question2']), axis=1)
# 
# print("trigram_Q1")
# test["trigram_Q1"] =  test.apply(lambda x: do_trigram(x['question1']), axis=1)
# print("trigram_Q2")
# test["trigram_Q2"] =  test.apply(lambda x: do_trigram(x['question2']), axis=1)
# 
# 
# #saving
# train['question1'] = train["trigram_Q1"]
# train['question2'] = train["trigram_Q2"]
# train = train.drop(['trigram_Q1', 'trigram_Q2'], axis=1)
# 
# 
# test['question1'] = test["trigram_Q1"]
# test['question2'] = test["trigram_Q2"]
# test = test.drop(['trigram_Q1', 'trigram_Q2'], axis=1)
# 
# 
#==============================================================================



train.to_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/train_spacy_cleaned.csv', index=False, encoding="utf-8")

test.to_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/test_spacy_cleaned.csv', index=False, encoding="utf-8")




#==============================================================================
# 
# sent = [u'the', u'new', u'york', u'times', u'is', u'a', u'newspaper']
# print(trigram_model[bigram_model[sent]])
# [u'the', u'new_york_times', u'is', u'a', u'newspaper']
# 
# 
#==============================================================================



#==============================================================================
# 
# questionsGensim =  [bigram_model[question.split(" ")] for question in questions]
# 
# bigram_model[u'What is the step by step guide to invest in share market in india?'] 
# 
# sent = questions[3].split(" ")
# print(bigram_model[sent])
# 
#==============================================================================

#==============================================================================
# sentence_stream = [num for num, doc in enumerate(questions) if type(doc)==float]
# print(type(questions[606131]))
#==============================================================================

#print(questions[606132])


#==============================================================================
# 
# from gensim.models import Phrases
# documents = ["the mayor of new york was there", "machine learning can be useful sometimes","new york mayor was present"]
# 
# sentence_stream = [doc.split(" ") for doc in documents]
# bigram = Phrases(sentence_stream, min_count=1, threshold=2)
# 
# sent = [u'the', u'mayor', u'of', u'new', u'york', u'was', u'there']
# print(bigram[sent])
# 
#==============================================================================












