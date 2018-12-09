# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 17:15:00 2018

@author: l00467141
"""

# In[]
from sklearn.feature_extraction.text import TfidfVectorizer

# In[]
vec = TfidfVectorizer()

# In[]
corpus = ['This is sample document.', 'another random document.', 'third sample document text']

# In[]
X = vec.fit_transform(corpus)#Learn vocabulary and idf, return term-document matrix

# In[]
print(vec.get_feature_names() )
# In[]
print(vec.get_params() )
# In[]
print(X)   #(#doc, #wordFeature)   weight


# In[]
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
svd = TruncatedSVD(2)

norm = Normalizer(copy=False)

lsa = make_pipeline(svd, norm)

X = vec.fit_transform(corpus)

X = lsa.fit_transform(X)

print(svd.explained_variance_ratio_.sum())
#0.576009049909   # if svd = TruncatedSVd(3), sum=1.0

print(svd.explained_variance_ratio_)  #每个变量能够解释的信息量（Variance）占比
#[ 0.02791594  0.54809311]

print(X)
'''
[[ 0.93630095 -0.35119871]
 [ 0.49995544  0.86605113]
 [ 0.93630095 -0.35119871]]
'''
print(svd.components_ ) #每个新特征中，原词汇权重的系数
'''[[ 0.23229736  0.51070605  0.31620157  0.23229736  0.4809589   0.31620157    0.31620157  0.31620157]
 [ 0.61930819  0.15015437 -0.18253737  0.61930819 -0.27764876 -0.18253737   -0.18253737 -0.18253737]'''