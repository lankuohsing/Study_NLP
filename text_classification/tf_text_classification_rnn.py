# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 16:15:42 2020

@author: lankuohsing
"""

import numpy as np
import os
import re
import tensorflow as tf
import string
import matplotlib.pyplot as plt
# In[]
def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)#转换为小写
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')#取出html的符号
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation), '')#去除标点
# In[]
dataset_dir = os.path.join("../dataset", 'aclImdb')
#os.listdir(dataset_dir)
train_dir = os.path.join(dataset_dir, 'train')
test_dir = os.path.join(dataset_dir, 'test')
BUFFER_SIZE = 10000
BATCH_SIZE = 64
#os.listdir(train_dir)
train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    train_dir, batch_size=BATCH_SIZE,shuffle=True)#
test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    test_dir, batch_size=BATCH_SIZE,shuffle=True)#

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
# In[]
VOCAB_SIZE=1000
sequence_length = 100
encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
        standardize=custom_standardization,
        max_tokens=VOCAB_SIZE,
        output_mode='int',
        output_sequence_length=sequence_length)#The tensors of indices are 0-padded to the longest sequence in the batch (unless you set a fixed output_sequence_length):
encoder.adapt(train_ds.map(lambda text, label: text))
# In[]
model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=16,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])
# In[]
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])
# In[]
history = model.fit(train_ds, epochs=10,
                    validation_data=test_ds,
                    validation_steps=30)
# In[]
test_loss, test_acc = model.evaluate(test_ds)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))
# In[]
plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
plot_graphs(history, 'accuracy')
plt.ylim(None,1)
plt.subplot(1,2,2)
plot_graphs(history, 'loss')
plt.ylim(0,None)