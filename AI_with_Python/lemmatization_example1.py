# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 20:20:22 2018

@author: l00467141
"""

# In[]
from nltk.stem import WordNetLemmatizer
# In[]
input_words = ['writing', 'calves', 'be', 'branded', 'horse', 'randomize',
               'possibly', 'provision', 'hospital', 'kept', 'scratchy', 'code']

# In[]
# Create lemmatizer object
lemmatizer = WordNetLemmatizer()
# Create a list of lemmatizer names for display
lemmatizer_names = ['NOUN LEMMATIZER', 'VERB LEMMATIZER']
formatted_text = '{:>24}' * (len(lemmatizer_names) + 1)
print('\n', formatted_text.format('INPUT WORD', *lemmatizer_names),'\n', '='*75)
# In[]
# Lemmatize each word and display the output
for word in input_words:
    output = [word, lemmatizer.lemmatize(word, pos='n'),
              lemmatizer.lemmatize(word, pos='v')]
    print(formatted_text.format(*output))