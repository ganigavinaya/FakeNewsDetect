#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 16:56:32 2017

@author: vinaya
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 18:53:04 2017

@author: shalin
"""

import spacy
from spacy.matcher import Matcher
from spacy.attrs import ORTH
import json
import nltk
from nltk.util import bigrams,trigrams
import pandas as pd
from tqdm import tqdm,trange
from textblob import TextBlob 
from textblob.sentiments import NaiveBayesAnalyzer
from gensim.models import ldamodel
from gensim import corpora
from nltk.stem.porter import PorterStemmer
import numpy as np
from sklearn.preprocessing import MinMaxScaler
p_stem = PorterStemmer()

parse = spacy.load('en_core_web_md')
nlp = spacy.load('en_core_web_lg')


dataset = json.load(open("C:/Users/shalin/Desktop/Fake_News_Detection/Stance_Detection/BaseLine/stories.json",encoding='utf-8'))
dataframe = pd.DataFrame(dataset)
dataframe = dataframe.drop(['date_added','language','processed','stanford','url'],1)
    

uci_dataframe = pd.read_csv("C:/Users/shalin/Desktop/Fake_News_Detection/Stance_Detection/BaseLine/uci.csv")
cols = [0,1,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
uci_dataframe = uci_dataframe.drop(uci_dataframe.columns[cols],axis=1)
uci_label = []
for i in range(39644): 
    uci_label.append('REAL')
uci_dataframe['Label'] = uci_label


fake_dataframe = pd.read_csv("C:/Users/shalin/Desktop/Fake_News_Detection/Stance_Detection/BaseLine/fake.csv")
cols = [1,2,3,6,7,8,9,10,11,13,14,15,16,17,18,19]
fake_dataframe = fake_dataframe.drop(fake_dataframe.columns[cols],axis=1)

def mod(num):
    if num >= 0:
        return num
    else:
        return num*-1

i = 0
for dict_id in tqdm(dataframe._id):
    dataframe._id[i]  = dict_id['$oid']
    i += 1

trainDataList = [] 
for i in trange(50000):
    tempdata = dataframe.iloc[i,:]
    body = tempdata[1]
    _id = tempdata[0]
    source = tempdata[3]
    title = tempdata[4]
    
    parsed_body = parse(body)
    
    body_nouns = [x.text for x in parsed_body if x.pos_ == 'NOUN' or x.pos_ == 'PROPN']
    noun_len = len(body_nouns)
    body_verbs = [x.text for x in parsed_body if x.pos_ == 'VERB']
    verb_len = len(body_verbs)
    
    appData = pd.Series([_id,source,body,title,noun_len,verb_len])
    trainDataList.append(appData)

trainDataFrame = pd.DataFrame(trainDataList)
trainDataFrame.columns = ['Id','Source','Body','Title','Body_Noun_No','Body_Verb_No']


title_token_count = []
for i in trange(50000):
    tempdata = trainDataFrame.iloc[i,:]
    title = tempdata[3]
    doc_title = parse(title)
    count = len(doc_title.count_by(ORTH))
    title_token_count.append(count)
trainDataFrame['n_tokens_title'] = title_token_count

body_token_count = []
for i in trange(50000):
    tempdata = trainDataFrame.iloc[i,:]
    body = tempdata[2]
    count = len(nltk.word_tokenize(body))
    body_token_count.append(count)
trainDataFrame['n_tokens_content'] = body_token_count

n_unique_tokens = []
for i in trange(50000):
    tempdata = trainDataFrame.iloc[i,:]
    body = tempdata[2]
    doc_body = parse(body)
    
    u_count = len(doc_body.count_by(ORTH))
    u_count_ratio = u_count/count
    n_unique_tokens.append(u_count_ratio)
trainDataFrame['n_unique_tokens'] = n_unique_tokens

n_non_stop_words = []
n_non_stop_unique_tokens = []
stopwords = nltk.corpus.stopwords.words('english')
for i in trange(50000):
    tempdata = trainDataFrame.iloc[i,:]
    body = tempdata[2]
    count = len(nltk.word_tokenize(body)) + 1
    non_stop_word_count = 0
    for token in nltk.word_tokenize(body):
        if token not in stopwords:
            non_stop_word_count+=1
    rat = non_stop_word_count/count
    n_non_stop_words.append(rat)
    
    doc = parse(body)
    u_non_s_w_count = 0
    for token in doc:
        if token.is_stop == False:
            u_non_s_w_count+=1
    rat = u_non_s_w_count/count
    n_non_stop_unique_tokens.append(rat)    
trainDataFrame['n_non_stop_words'] = n_non_stop_words
trainDataFrame['n_non_stop_unique_tokens'] = n_non_stop_unique_tokens

average_token_length = []
for i in trange(50000):
    tempdata = trainDataFrame.iloc[i,:]
    body = tempdata[2]
    count = len(nltk.word_tokenize(body)) + 1
    sum = 0
    for word in nltk.word_tokenize(body):
        sum += len(word)
    avg = sum/count
    average_token_length.append(avg)
trainDataFrame['average_token_length'] = average_token_length

#using original dataframe for dates

#LDA - Latent Dirichlet Allocation using gensim. No of topics are 5 for now....
corpus = []
for i in trange(50000):
    tempdata = trainDataFrame.iloc[i,:]
    body = tempdata[2]
    corpus.append(body)
    
token_corpus = []
for doc in tqdm(corpus):
    raw = doc.lower()
    tokens = nltk.word_tokenize(raw)
    tokens_ws = []
    for t in tokens:
        if t not in stopwords:
            tokens_ws.append(t)
    token_stem = []
    for t in tokens_ws:
        token_stem.append(p_stem.stem(t))        
    token_corpus.append(token_stem)    

dictionary = corpora.Dictionary(token_corpus)
bow = [dictionary.doc2bow(doc) for doc in token_corpus]

ldamodel = ldamodel.LdaModel(bow,num_topics=5,id2word=dictionary,passes=2)

lda = []
for doc in tqdm(bow):
    l = ldamodel[doc]
    ldadata = {}
    for tup in l:
        ldadata[tup[0]] = tup[1]    
    lda.append(ldadata)     

lda_dataframe = pd.DataFrame(lda)
lda_dataframe.columns = ['LDA_00','LDA_01','LDA_02','LDA_03','LDA_04']
templist = ['LDA_00','LDA_01','LDA_02','LDA_03','LDA_04']
for t in templist:
    trainDataFrame[t] = pd.Series(lda_dataframe[t])
  
#Sentimental Analysis
text_sentiment_polarity = []
for i in trange(50000):
    tempdata = trainDataFrame.iloc[i,:]
    body = tempdata[2]
    blob = TextBlob(str(body))
    text_sentiment_polarity.append(blob.sentiment.polarity)        
trainDataFrame['text_sentiment_polarity'] = text_sentiment_polarity

text_subjectivity = []
for i in trange(50000):
    tempdata = trainDataFrame.iloc[i,:]
    body = tempdata[2]
    blob = TextBlob(str(body))
    text_subjectivity.append(blob.sentiment.subjectivity)        
trainDataFrame['text_subjectivity'] = text_subjectivity

#rate of positive/negative words    
rate_of_positive_words = []
rate_of_negative_words = []
rate_of_positive_words_nn = []
rate_of_negative_words_nn = []
avg_pos = []
min_pos = []
max_pos = []
avg_neg = []
min_neg = []
max_neg = []

for i in trange(50000):
    n_pos=1
    n_neg=1
    n_neu=1
    count=1
    sum_pos_polarity=0
    sum_neg_polarity=0
    max_pos_polarity=[0.0000]
    min_pos_polarity=[99.0000]
    max_neg_polarity=[-99.000]
    min_neg_polarity=[0.0000]
    
    tempdata = trainDataFrame.iloc[i,:]
    body_tokens = nltk.word_tokenize(str(tempdata[2]))
    count = len(body_tokens)+1
    for word in  body_tokens:
        blob = TextBlob(word)
        pol = blob.sentiment.polarity
        if pol > 0:
            sum_pos_polarity+=pol
            max_pos_polarity.append(pol)
            min_pos_polarity.append(pol)
            n_pos+=1
        if pol < 0:
            sum_neg_polarity+=pol
            max_neg_polarity.append(pol)
            min_neg_polarity.append(pol)
            n_neg+=1
        else:
            n_neu+=1
    
    no_nn = (count-n_neu) + 1 
    r_pos = n_pos/count
    r_neg = n_neg/count
    r_pos_non_neu = n_pos/no_nn
    r_neg_non_neu = n_neg/no_nn
    avg_pos_polarity = sum_pos_polarity/n_pos
    avg_neg_polarity = sum_neg_polarity/n_neg
    max_pos_pol_index = np.argmax(max_pos_polarity)
    max_pos_polarity1 = max_pos_polarity[max_pos_pol_index]
    min_pos_pol_index = np.argmin(min_pos_polarity)
    min_pos_polarity1 = min_pos_polarity[min_pos_pol_index]
    max_neg_pol_index = np.argmax(max_neg_polarity)
    max_neg_polarity1 = max_neg_polarity[max_neg_pol_index]
    min_neg_pol_index = np.argmin(min_neg_polarity)
    min_neg_polarity1 = min_neg_polarity[min_neg_pol_index]
    
    rate_of_positive_words.append(r_pos)
    rate_of_negative_words.append(r_neg)
    rate_of_positive_words_nn.append(r_pos_non_neu)
    rate_of_negative_words_nn.append(r_neg_non_neu)
    avg_pos.append(avg_pos_polarity)
    max_pos.append(max_pos_polarity1)
    min_pos.append(min_pos_polarity1)
    avg_neg.append(avg_neg_polarity)
    max_neg.append(max_neg_polarity1)
    min_neg.append(min_neg_polarity1)

trainDataFrame['rate_of_positive_words'] = rate_of_positive_words
trainDataFrame['rate_of_negative_words'] = rate_of_negative_words
trainDataFrame['rate_of_positive_words_nn'] = rate_of_positive_words_nn
trainDataFrame['rate_of_negative_words_nn'] = rate_of_negative_words_nn
trainDataFrame['avg_positive_polarity'] = avg_pos
trainDataFrame['max_positive_polarity'] = max_pos
trainDataFrame['min_positive_polarity'] = min_pos
trainDataFrame['avg_negative_polarity'] = avg_neg
trainDataFrame['max_negative_polarity'] = max_neg
trainDataFrame['min_negative_polarity'] = min_neg
   

title_subjectivity = []
for i in trange(50000):
    tempdata = trainDataFrame.iloc[i,:]
    title = tempdata[3]
    blob = TextBlob(str(title))
    title_subjectivity.append(blob.sentiment.subjectivity)        
trainDataFrame['title_subjectivity'] = title_subjectivity

title_sentiment_polarity = []
for i in trange(50000):
    tempdata = trainDataFrame.iloc[i,:]
    title = tempdata[3]
    blob = TextBlob(str(title))
    title_sentiment_polarity.append(blob.sentiment.polarity)        
trainDataFrame['title_sentiment_polarity'] = title_sentiment_polarity

title_abs_sentiment_polarity = []
for i in trange(50000):
    tempdata = trainDataFrame.iloc[i,:]
    title = tempdata[3]
    blob = TextBlob(str(title))
    title_abs_sentiment_polarity.append(mod(blob.sentiment.polarity))        
trainDataFrame['title_abs_sentiment_polarity'] = title_abs_sentiment_polarity
 
trainDataFrame.to_csv("C:/Users/shalin/Desktop/Fake_News_Detection/Stance_Detection/BaseLine/temp.csv",encoding='utf-8')

##N-grams - Generate N-grams bow for each document body over entire corpus
corpus = []
for i in trange(50000):
    tempdata = trainDataFrame.iloc[i,:]
    body = str(tempdata[2]).lower()
    token_body = nltk.word_tokenize(body)
    token_body_ws = []
    for token in token_body:
        if token not in stopwords:
            token_body_ws.append(token)
    
    final_token_body = []
    for token in token_body_ws:
        final_token_body.append(p_stem.stem(token))
    corpus.append(final_token_body)   

#ARI - Automated Readability Index
    
trainDataFrame = pd.read_csv("C:/Users/shalin/Desktop/Fake_News_Detection/Stance_Detection/BaseLine/temp.csv",encoding='utf-8')   
trainDataFrame = trainDataFrame.drop(['Unnamed: 0'],1)

ari = []
for i in trange(50000):
    tempdata = trainDataFrame.iloc[i,:]
    body = tempdata[2]
    no_of_char = 0
    no_of_words = 0
    no_of_sents = 0
    
    no_of_words = len(nltk.word_tokenize(str(body)))
    no_of_sents = len(nltk.sent_tokenize(str(body)))
    for token in nltk.word_tokenize(str(body)):
        no_of_char += len(token)
        
    if no_of_sents != 0 and no_of_words != 0:
        doc_ari = 4.71*(no_of_char/no_of_words) + 0.5*(no_of_words/no_of_sents) - 21.43
    else:
        doc_ari = -21.43
    ari.append(doc_ari)
    
trainDataFrame['ARI'] = ari    

trainDataFrame.to_csv("C:/Users/shalin/Desktop/Fake_News_Detection/Stance_Detection/BaseLine/scaled_temp.csv",encoding='utf-8')


#Unigrams - Bigrams - Trigrams

p_arti1 = json.load(open("C:/Users/shalin/Desktop/Fake_News_Detection/Stance_Detection/BaseLine/processed_articles1.json",encoding='utf-8'))
p_arti2 = json.load(open("C:/Users/shalin/Desktop/Fake_News_Detection/Stance_Detection/BaseLine/processed_articles2.json",encoding='utf-8'))
p_arti3 = json.load(open("C:/Users/shalin/Desktop/Fake_News_Detection/Stance_Detection/BaseLine/processed_articles3.json",encoding='utf-8'))

temp_corpus = []
for d in p_arti1:
    temp_corpus.append(d['body'])
for d in p_arti2:
    temp_corpus.append(d['body'])
for d in p_arti3:
    temp_corpus.append(d['body'])

corpus_unigrams = []
for doc in tqdm(temp_corpus):
    corpus = []
    for sent in doc:
        for word in sent:
            corpus.append(word)
    corpus_unigrams.append(corpus)
trainDataFrame['corpus_unigrams'] = corpus_unigrams   

corpus_bigrams = []
for doc in tqdm(corpus_unigrams):
    corpus_bigrams.append(list(bigrams(doc)))
trainDataFrame['corpus_bigrams'] = corpus_bigrams   

corpus_trigrams = []
for doc in tqdm(corpus_unigrams):
    corpus_trigrams.append(list(trigrams(doc)))
trainDataFrame['corpus_trigrams'] = corpus_trigrams   

#Total counts
uni_total = {}
for doc in tqdm(corpus_unigrams):
    for token in doc:
        if token in uni_total:
            uni_total[token]+=1
        else:
            uni_total[token]=1

bi_total = {}
for doc in tqdm(corpus_bigrams):
    for token in doc:
        if token in bi_total:
            bi_total[token]+=1
        else:
            bi_total[token]=1

tri_total = {}
for doc in tqdm(corpus_trigrams):
    for token in doc:
        if token in tri_total:
            tri_total[token]+=1
        else:
            tri_total[token]=1

#Prob calculations......
uni_prob = []
bi_prob = []
tri_prob = []
for i in trange(50000):
    appdata = trainDataFrame.iloc[i,:]
    unigrams = appdata[33]
    bigrams = appdata[34]
    trigrams = appdata[35]
    
    uni_dict = {}
    for token in unigrams:
        if token in uni_dict:
            uni_dict[token]+=1
        else:
            uni_dict[token]=1
    
    prob_uni_dict = {}
    for token in uni_dict.keys():
        count = uni_dict[token]
        total_count = uni_total[token]
        if total_count != 0:
            prob = count/total_count
        else:
            prob = 0
        prob_uni_dict[token] = prob    
        
    uni_prob.append(prob_uni_dict)

    bi_dict = {}
    for token in bigrams:
        if token in bi_dict:
            bi_dict[token]+=1
        else:
            bi_dict[token]=1
    
    prob_bi_dict = {}
    for token in bi_dict.keys():
        count = bi_dict[token]
        total_count = bi_total[token]
        if total_count != 0:
            prob = count/total_count
        else:
            prob = 0
        prob_bi_dict[token] = prob    
        
    bi_prob.append(prob_bi_dict)

    tri_dict = {}
    for token in trigrams:
        if token in tri_dict:
            tri_dict[token]+=1
        else:
            tri_dict[token]=1
    
    prob_tri_dict = {}
    for token in tri_dict.keys():
        count = tri_dict[token]
        total_count = tri_total[token]
        if total_count != 0:
            prob = count/total_count
        else:
            prob = 0
        prob_tri_dict[token] = prob    
        
    tri_prob.append(prob_tri_dict)

trainDataFrame['corpus_unigrams'] = uni_prob   
trainDataFrame['corpus_bigrams'] = bi_prob   
trainDataFrame['corpus_trigrams'] = tri_prob   

trainDataFrame = pd.read_csv("C:/Users/shalin/Desktop/Fake_News_Detection/Stance_Detection/BaseLine/temp.csv",encoding='utf-8')
trainDataFrame = trainDataFrame.drop(['Unnamed: 0'],1)

#Scaling data for keras NN
min_max_scaler = MinMaxScaler()
trainDataFrame.iloc[:,4] = min_max_scaler.fit_transform(np.array(trainDataFrame.iloc[:,4]).reshape(-1,1))
trainDataFrame.iloc[:,5] = min_max_scaler.fit_transform(np.array(trainDataFrame.iloc[:,5]).reshape(-1,1))
trainDataFrame.iloc[:,6] = min_max_scaler.fit_transform(np.array(trainDataFrame.iloc[:,6]).reshape(-1,1))
trainDataFrame.iloc[:,7] = min_max_scaler.fit_transform(np.array(trainDataFrame.iloc[:,7]).reshape(-1,1))
trainDataFrame.iloc[:,8] = min_max_scaler.fit_transform(np.array(trainDataFrame.iloc[:,8]).reshape(-1,1))
trainDataFrame.iloc[:,10] = min_max_scaler.fit_transform(np.array(trainDataFrame.iloc[:,10]).reshape(-1,1))
trainDataFrame.iloc[:,11] = min_max_scaler.fit_transform(np.array(trainDataFrame.iloc[:,11]).reshape(-1,1))
trainDataFrame.iloc[:,21] = min_max_scaler.fit_transform(np.array(trainDataFrame.iloc[:,21]).reshape(-1,1))
trainDataFrame.iloc[:,32] = min_max_scaler.fit_transform(np.array(trainDataFrame.iloc[:,32]).reshape(-1,1))
