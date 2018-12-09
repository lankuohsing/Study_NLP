# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 19:34:56 2018

@author: l00467141
"""
# In[]
from nltk.tokenize import sent_tokenize, word_tokenize, WordPunctTokenizer


# In[]
# Define input text
input_text = "Do you know how tokenization works? It's actually quiteinteresting! Let's analyze a couple of sentences and figure it out."
# In[]
# Sentence tokenizer
print("\nSentence tokenizer:")
print(sent_tokenize(input_text))
# Word tokenizer
print("\nWord tokenizer:")
print(word_tokenize(input_text))
# WordPunct tokenizer
print("\nWord punct tokenizer:")
print(WordPunctTokenizer().tokenize(input_text))