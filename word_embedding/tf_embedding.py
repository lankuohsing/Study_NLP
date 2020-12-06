# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 22:51:50 2020

@author: lankuohsing
"""

import io
import os
import re
import shutil
import string
import tensorflow as tf#tf2.3

from datetime import datetime
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Activation, Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# In[]
dataset_dir = os.path.join("dataset", 'aclImdb')
#os.listdir(dataset_dir)

train_dir = os.path.join(dataset_dir, 'train')
#os.listdir(train_dir)
# In[]
batch_size = 1024
seed = 123
train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    train_dir, batch_size=batch_size, validation_split=0.2,
    subset='training', seed=seed)#label_mode='int';return (texts, labels):(batch_size,) (batch_size,)
val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    train_dir, batch_size=batch_size, validation_split=0.2,
    subset='validation', seed=seed)
# In[]
#for text_batch, label_batch in train_ds.take(1):
#  for i in range(5):
#    print(label_batch[i].numpy(), text_batch.numpy()[i])

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# In[]
# Create a custom standardization function to strip HTML break tags '<br />'.
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)#转换为小写
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')#取出html的符号
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation), '')#去除标点

# Vocabulary size and number of words in a sequence.
vocab_size = 10000
sequence_length = 100

# Use the text vectorization layer to normalize, split, and map strings to
# integers. Note that the layer uses the custom standardization defined above.
# Set maximum_sequence length as all samples are not of the same length.
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)

# Make a text-only dataset (no labels) and call adapt to build the vocabulary.
text_ds = train_ds.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)
# In[]
embedding_dim=16

model = Sequential([
  vectorize_layer,
  Embedding(vocab_size, embedding_dim, name="embedding"),
  GlobalAveragePooling1D(),#对一句话里面的所有单词的Word Embedding向量求平均
  Dense(16, activation='relu'),#隐藏层
  Dense(1)#输出层，输出两类
])
# In[]
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
# In[]
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
# In[]
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[tensorboard_callback])
# In[]
model.summary()

# In[]
weights = model.get_layer('embedding').get_weights()[0]
vocab = vectorize_layer.get_vocabulary()
# In[]
out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

for index, word in enumerate(vocab):
  if  index == 0: continue # skip 0, it's padding.
  vec = weights[index]
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
  out_m.write(word + "\n")
out_v.close()
out_m.close()
# In[]
model.save('./models')