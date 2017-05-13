# -*- coding: utf-8 -*-
"""
helper for quoraScapy
"""
import csv
import codecs
import numpy as np
import pandas as pd
import spacy
nlp = spacy.load('en')
np.set_printoptions(threshold=400000)
#import itertools as it
from os.path import isfile
from collections import Counter
from pprint import pprint
from itertools import islice

from gensim.models import KeyedVectors
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Bidirectional, GRU, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint


def preprocess_training_data(train_data, processed_train_data):
    print('Processing training text dataset')
    texts_1 = [] 
    texts_2 = []
    labels = []
    with codecs.open(train_data, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        for values in reader:
            texts_1.append((values[3]))
            texts_2.append((values[4]))
            labels.append(int(values[5]))
    print('Found %s texts in train.csv' % len(texts_1))
    print('Ex of text: ',texts_1[:5])
    trainingPreprocessed = pd.DataFrame({'q1':texts_1, 'q2':texts_2, 'label':labels})
    trainingPreprocessed.to_csv(processed_train_data, index=False, encoding="utf-8")
    print('Training Data Preprocessed')
    return texts_1, texts_2, labels


def preprocess_test_data(test_data, processed_test_data):
    print('Processing test text dataset')
    test_texts_1 = []
    test_texts_2 = []
    test_ids = []
    with codecs.open(test_data, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        for values in reader:
            test_texts_1.append((values[1]))
            test_texts_2.append((values[2]))
            test_ids.append(values[0])
    print('Found %s texts in test.csv' % len(test_texts_1))
    print('Ex of test text: ',test_texts_1[:5])
    testPreprocessed = pd.DataFrame({'q1':test_texts_1, 'q2':test_texts_2, 'id':test_ids})
    testPreprocessed.to_csv(processed_test_data, index=False, encoding="utf-8")
    print('Test Data Preprocessed')
    return test_texts_1, test_texts_2, test_ids
    

def loading_training_data(processed_train_data):
    print('Loading processed training text dataset')
    texts_1 = [] 
    texts_2 = []
    labels = []
    with codecs.open(processed_train_data, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        for values in reader:
            texts_1.append(values[1])
            texts_2.append(values[2])
            labels.append(int(values[0]))
    print('Found %s texts in train.csv' % len(texts_1))
    print('Ex of text: ',texts_1[:5])
    return texts_1, texts_2, labels
    

def loading_test_data(processed_train_data):
    print('Loading processed test text dataset')
    test_texts_1 = [] 
    test_texts_2 = []
    test_ids = []
    with codecs.open(processed_train_data, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        for values in reader:
            test_texts_1.append(values[1])
            test_texts_2.append(values[2])
            test_ids.append(int(values[0]))
    print('Found %s texts in test.csv' % len(test_texts_1))
    print('Ex of text: ',test_texts_1[:5])
    return test_texts_1, test_texts_2, test_ids

def dummyFirst(textNLP):
    firstWord = textNLP[0].lemma_
    countWhy = 0
    countHow = 0
    countWhat = 0
    countIf = 0
    countWhich = 0
    countWho = 0
    countCan = 0
    countIs = 0
    countHave = 0
    countDo = 0
    countWill = 0    
    countShould = 0    
    countWhen = 0  
    countWhere = 0
    if(firstWord == "why"):
        countWhy +=1
    elif(firstWord == "how"):
        countHow +=1    
    elif(firstWord == "what"):
        countWhat +=1  
    elif(firstWord == "if"):
        countIf +=1  
    elif(firstWord == "which"):
        countWhich +=1  
    elif(firstWord == "who"):
        countWho +=1  
    elif(firstWord == "should"):
        countShould +=1  
    elif(firstWord == "when"):
        countWhen +=1  
    elif(firstWord == "where"):
        countWhere +=1  
    elif(firstWord == "can" or firstWord == "could"):
        countCan +=1 
    elif(firstWord == "is" or firstWord == "are"):
        countIs +=1  
    elif(firstWord == "has" or firstWord == "have"):
        countHave +=1  
    elif(firstWord == "do" or firstWord == "does"):
        countDo +=1 
    elif(firstWord == "will" or firstWord == "would"):
        countWill +=1    
    return [countWhy, countHow, countWhat, countIf, countWhich, countWho, countCan, countIs, countHave, countDo, countWill, countWhere, countWhen, countShould]

def countSentences(textNLP):
    numMax = 1
    for num, sentence in enumerate(textNLP.sents):
        numMax = max(numMax, num+1)
    return numMax

def countEntities(textNLP):
    countPerson   = 0
    countNorp     = 0
    countFacility = 0
    countOrg      = 0
    countGpe      = 0
    countLoc      = 0
    countProduct  = 0
    countEvent    = 0
    countWord_of_art = 0
    countLanguage = 0
    for num, entity in enumerate(textNLP.ents):
        if(entity.label_ == "PERSON"):
            countPerson +=1
        elif(entity.label_ == "NORP"):
            countNorp +=1    
        elif(entity.label_ == "FACILITY"):
            countFacility +=1  
        elif(entity.label_ == "ORG"):
            countOrg +=1  
        elif(entity.label_ == "GPE"):
            countGpe +=1  
        elif(entity.label_ == "LOC"):
            countLoc +=1  
        elif(entity.label_ == "PRODUCT"):
            countProduct +=1  
        elif(entity.label_ == "EVENT"):
            countEvent +=1  
        elif(entity.label_ == "WORK_OF_ART"):
            countWord_of_art +=1  
        elif(entity.label_ == "LANGUAGE"):
            countLanguage +=1  
    return [countPerson, countNorp, countFacility, countOrg, countGpe, countLoc, countProduct, countEvent, countWord_of_art, countLanguage]

def countNounPronounVerb(textNLP):
    countNoun  = 0
    countPropn = 0
    countVerb  = 0
    countPron  = 0
    for token in textNLP:
        if(token.pos_ == "NOUN"):
            countNoun +=1
        elif(token.pos_== "PROPN"):
            countPropn +=1    
        elif(token.pos_ == "VERB"):
            countVerb +=1  
        elif(token.pos_ == "PRON"):
            countPron +=1  
    return [countNoun, countPropn, countVerb, countPron]
        


def scapyCounts(text):
    output = []
    #for num, text in islice(enumerate(text),0,20):
    for num, text in enumerate(text):
        textNLP = nlp(text)
        outputQuestion = []
        
        #Sentences count
        sentenceCount = countSentences(textNLP)
        
        #Entity label_ count
        entitiesCount = countEntities(textNLP)
        
        #Pos_ count
        posCount = countNounPronounVerb(textNLP)
        
        #dummy first word
        #firstWordDummy = dummyFirst(textNLP)

        #char count
        charCount = len(text)
        
        #word count
        wordCount = len(text.split())

        #output for this Quora Question
        outputQuestion.append(sentenceCount)
        outputQuestion.extend(entitiesCount)
        outputQuestion.extend(posCount)
        #outputQuestion.extend(firstWordDummy)
        outputQuestion.append(charCount)
        outputQuestion.append(wordCount)

        output.append(outputQuestion)

        if((num) % 10000 == 0):
            print("scapyCounts Doing ID:", num)
            
    return np.array(output)

def wordCompare(text1, text2):
    output = []
    #for num, text in islice(enumerate(text1),0,20):
    for num, text in enumerate(text1):
        outputQuestion = []
        
        #flag if same first word
        sameFirstWord = 0
        if (text == ""):
            if (text2[0] == ""):
                sameFirstWord = 1
            else:
                sameFirstWord = 0
        elif(text2[0] == ""):
            sameFirstWord = 0
        elif(text[0].lower == text2[0][0].lower):
            sameFirstWord = 1
        
        #wordCountDiff
        wordCount1 = len(text.split())
        wordCount2 = len(text2[num].split())
        wordCountDiff = wordCount1 - wordCount2
        
        #char countDiff
        charCount1 = len(text)
        charCount2 = len(text2[num])
        charCountDiff = charCount1 - charCount2
        
        outputQuestion.append(sameFirstWord)
        outputQuestion.append(wordCountDiff)
        outputQuestion.append(charCountDiff)
        output.append(outputQuestion)
        

        if((num) % 10000 == 0):
            print("scapyCompare Doing ID:", num)
            
    return np.array(output)





def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    punctuation_to_token = {}
    punctuation_to_token["!"] = '||EXCLAMATIONMARK||'
    punctuation_to_token['"'] = '||QUOTATIONMARK||'
    punctuation_to_token["#"] = '||POUND||'
    punctuation_to_token["$"] = '||DOLAR||'
    punctuation_to_token["%"] = '||PERCENTAGE||'
    punctuation_to_token["&"] = '||AMPERSAND||'
    punctuation_to_token["'"] = '||APOSTROPHE||'
    punctuation_to_token["("] = '||OPENNINGPARENTHESIS||'
    punctuation_to_token[")"] = '||CLOSINGPARENTHESIS||'
    punctuation_to_token["*"] = '||STAR||'
    punctuation_to_token["+"] = '||PLUS||'
    punctuation_to_token[","] = '||COMMA||'
    punctuation_to_token["--"]= '||HYPHENS||'
    punctuation_to_token["-"] = '||DASH||'
    punctuation_to_token["."] = '||PERIOD||'
    punctuation_to_token["/"] = '||SLASH||'
    punctuation_to_token[":"] = '||COLON||'
    punctuation_to_token[";"] = '||SEMICOLON||'
    punctuation_to_token["<"] = '||LOWERTHAN||'
    punctuation_to_token["="] = '||EQUAL||'
    punctuation_to_token[">"] = '||GREATERTHAN||'
    punctuation_to_token["?"] = '||QUESTIONMARK||'
    punctuation_to_token["@"] = '||AT||'
    punctuation_to_token["["] = '||OPENINGCROCHET||'
    punctuation_to_token["\\"]= '||ANTI_SLASH||'
    punctuation_to_token["]"] = '||CLOSINGCROCHET| '
    punctuation_to_token["^"] = '||CIRCONFLEX||'
    punctuation_to_token["_"] = '||UNDERSCORE||'
    punctuation_to_token["`"] = '||OTHER_APOSTROPHE||'
    punctuation_to_token["{"] = '||OPENINGACCOLADE||'
    punctuation_to_token["}"] = '||CLOSINGACCOLADE||'
    punctuation_to_token["~"] = '||TILDE||'
    punctuation_to_token["\n"] ='||NEWLINE||'
    #punctuation_to_token["|"] = '||STRAIGHTBAR||'
    return punctuation_to_token


def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    import re
    from nltk.corpus import stopwords
    from nltk.stem import SnowballStemmer

    # Convert words to lower case and split them
    text = text.lower()

    # Optionally, remove stop words
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)

    # Clean the text
    #text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    #text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    #text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" e.g.", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" b.g.", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r" u.s.", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r" 9-11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"j.k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    #punctuation tokens
    token_dict = token_lookup()
    for key, token in token_dict.items():
        text = text.replace(key, ' {} '.format(token))
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)

















































