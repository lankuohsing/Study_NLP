# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 22:43:35 2018

@author: l00467141
"""

from sklearn.feature_extraction.text import CountVectorizer

texts=["cat cat cat cat cat cat dog cat fish dog cat fish dog cat fish dog cat fish",
       "dog cat cat dog cat fish","fish bird dog cat fish", "bird dog cat fish"," dog cat fish"," dog cat fish"]
cv = CountVectorizer(min_df=1,max_df=5)#创建词袋数据结构
#cv = CountVectorizer()#创建词袋数据结构
cv_fit=cv.fit_transform(texts)

print("feature names: ",cv.get_feature_names())
print("dictionary: ",cv.vocabulary_)
#['bird', 'cat', 'dog', 'fish']
print(cv_fit.toarray()) #.toarray() 是将结果转化为稀疏矩阵矩阵的表示方式；
#[[0 1 1 1]
# [0 2 1 0]
# [1 0 0 1]
# [1 0 0 0]]
#每个词在所有文档中的词频
print(cv_fit.toarray().sum(axis=0))
#[2 3 2 2]