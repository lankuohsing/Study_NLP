# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 23:24:33 2020

@author: lankuohsing
"""

import load_data
import explore_data
import vectorize_data
# In[]
((train_texts, train_labels),(test_texts,test_labels))=load_data.load_imdb_sentiment_analysis_dataset("./dataset")



# In[]
midian_words_per_sample=explore_data.get_num_words_per_sample(train_texts)
# In[]
"""
1. Calculate the number of samples/number of words per sample ratio.
2. If this ratio is less than 1500, tokenize the text as n-grams and use a
simple multi-layer perceptron (MLP) model to classify them (left branch in the
flowchart below):
  a. Split the samples into word n-grams; convert the n-grams into vectors.
  b. Score the importance of the vectors and then select the top 20K using the scores.
  c. Build an MLP model.
3. If the ratio is greater than 1500, tokenize the text as sequences and use a
   sepCNN model to classify them (right branch in the flowchart below):
  a. Split the samples into words; select the top 20K words based on their frequency.
  b. Convert the samples into word sequence vectors.
  c. If the original number of samples/number of words per sample ratio is less
     than 15K, using a fine-tuned pre-trained embedding with the sepCNN
     model will likely provide the best results.
4. Measure the model performance with different hyperparameter values to find
   the best model configuration for the dataset.
"""
# In[]
S_W=len(train_texts)/midian_words_per_sample

# In[]
x_train, x_val=vectorize_data.ngram_vectorize(train_texts, train_labels, test_texts)
