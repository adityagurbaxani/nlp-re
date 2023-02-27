#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 22:52:50 2023

@author: agurbaxani
"""

import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
#nltk.download('all')

# punctuations
punctuations=list(string.punctuation)
#Using the stop words set
stop_words = set(stopwords.words('english'))

#Map for Parts-of-Speech (POS) tags
tag_map = {
    'NN': 'n', #Nouns
    'VB': 'v', #Verbs
    'JJ': 'a', #Adjectives
    'RB': 'r'  #Adverbs
    }

#Using Lemmatizer to remove redundancy
wnl = WordNetLemmatizer()

#Defining the grammar and parser for Named Entity Recognition (NER)
grammar = r"""Chunk: {<RB.?>*<VB.?>*<NNP>*<CD>?<NN>?<IN>*<NN>+}"""
parser_chunking = nltk.RegexpParser(grammar)

#Reading the text data from file
lines = []
with open('./Data/nfr.txt') as f:
    lines = f.readlines()

    
#Create a DataFrame to process the text
df = pd.DataFrame()
df['Raw'] = lines


#Split each requirement into req-label & req
req_labels = []
req_texts = []
chunk_counts = []
chunks_as_text = []
keyword_count = []
keywords_as_text = []


#Master list of all keywords in Corpus
kw_master_list = []

for i in range(df.shape[0]):
    req = df.iloc[i,0]
    sep_index = req.find(":")
    req_labels.append(req[:sep_index])
    req_texts.append(req[sep_index+1:-1])
    #Converting text to lower case
    text = req[sep_index+1:-1].lower()
    #Tokenizing the requirements text
    words_in_text = word_tokenize(text)
    text_pos_tags = nltk.pos_tag(words_in_text)
    filtered_lemmatized = {True}
    for w,tag in text_pos_tags:
        lw = ''
        if(tag[:2] in tag_map):
            lw = wnl.lemmatize(w,tag_map[tag[:2]])
        else:
            lw = wnl.lemmatize(w)
        if(lw not in stop_words):
            if (lw not in punctuations):
                filtered_lemmatized.add(lw)
                kw_master_list.append(lw)
    filtered_lemmatized.remove(True)
    #Join the keywords as single string for DF
    keyword_count.append(len(filtered_lemmatized))
    keywords_as_text.append(''.join(w+', ' for w in filtered_lemmatized)[:-2])
    chunk_tree = parser_chunking.parse(text_pos_tags)
    chunks = []
    for ch in chunk_tree:
        if hasattr(ch, 'label'):
            chunks.append(''.join(c[0]+' ' for c in ch)[:-1])
    chunk_counts.append(len(chunks))
    chunks_as_text.append(''.join(ch+', ' for ch in chunks)[:-2])
            
    

df['req-label'] = req_labels
df['req'] = req_texts
df['chunk-count'] = chunk_counts
df['chunks-in-req'] = chunks_as_text
df['keyword-count'] = keyword_count
df['keywords-in-req'] = keywords_as_text

#Write DataFrame to CSV
df.to_csv('./output.csv')

#remove all words having length less than 3 from master list
kw_master_list = [w for w in kw_master_list if len(w) > 2]
fdist = FreqDist(kw_master_list)

#Frequency Distribution Plot
plt.figure(figsize = (16, 10), facecolor = None)
fdist.plot(10,cumulative=False)
plt.figure(figsize = (16, 10), facecolor = None)
fdist.plot(15,cumulative=False)
plt.figure(figsize = (16, 10), facecolor = None)
fdist.plot(20,cumulative=False)
plt.figure(figsize = (16, 10), facecolor = None)
fdist.plot(25,cumulative=False)

#print(fdist.most_common(20))
freq = pd.DataFrame(fdist.most_common(20))
freq.to_csv('./most-comman.csv')
freq.columns = ['keyword','count']
freq.sort_values(by='count',ascending=True,axis=0).plot.barh(y = 'count', x = 'keyword',figsize = (16, 10))

#Generating a word cloud
wc = WordCloud(width=1100, height=1100,
               background_color='white',
               stopwords=set(STOPWORDS),
               min_font_size=10).generate(''.join(kw+' ' for kw in kw_master_list))
plt.figure(figsize = (11, 11), facecolor = None)
plt.imshow(wc)
plt.axis("off")
plt.tight_layout(pad = 0)


#Considering single requirement for testing
temp = req_texts[1].lower()

#Tokenizing the requirements text
words_in_temp = word_tokenize(temp)


#Using POS Tags
temp_pos_tags = nltk.pos_tag(words_in_temp)
#tree = nltk.ne_chunk(temp_pos_tags)

#Using Lemmatizer to remove redundancy
wnl = WordNetLemmatizer()

#filtered_lemmatized = [wnl.lemmatize(w) for w, tag in temp_pos_tags]
filtered_lemmatized = []
for w,tag in temp_pos_tags:
    lw = ''
    if(tag[:2] in tag_map):
        lw = wnl.lemmatize(w,tag_map[tag[:2]])
    else:
        lw = wnl.lemmatize(w)
    if(lw not in stop_words):
        filtered_lemmatized.append(lw)


#grammar = "NP:{<MD>?<JJ>*<NN>}"
#grammar = "VP:{<MD>?<VB>}"
tree_chunk = parser_chunking.parse(temp_pos_tags)
for chunk in tree_chunk:
    if hasattr(chunk, 'label'):
        print(''.join(c[0]+' ' for c in chunk))







