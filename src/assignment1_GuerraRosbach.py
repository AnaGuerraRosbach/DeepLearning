# -*- coding: utf-8 -*-
"""
   Deep Learning for NLP
   Assignment 1: Sentiment Classification on a Feed-Forward Neural Network using Pretrained Embeddings
   Remember to use PyTorch for your NN implementation.
   Original code by Hande Celikkanat & Miikka Silfverberg. Minor modifications by Sharid Loáiciga.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gensim
import os
#from allennlp.nn.util import get_text_field_mask

# Add the path to these data manipulation files if necessary:
# import sys
# sys.path.append('</PATH/TO/DATA/MANIP/FILES>')
from data_semeval import *
from paths import data_dir, model_dir


# name of the embeddings file to use
# Alternatively, you can also use the text file GoogleNews-pruned2tweets.txt (from Moodle),
# or the full set, wiz. GoogleNews-vectors-negative300.bin (from https://code.google.com/archive/p/word2vec/) 
embeddings_file = 'GoogleNews-pruned2tweets.bin'


#--- hyperparameters ---

# Feel free to experiment with different hyperparameters to see how they compare! 
# You can turn in your assignment with the best settings you find.

n_classes = len(LABEL_INDICES)
n_epochs = 30 
learning_rate = 0.00025
hidden_layer_1 = 128
hidden_layer_2 = 64
report_every = 1
verbose = False



#--- auxilary functions ---

# To convert string label to pytorch format:
def label_to_idx(label):
    return torch.LongTensor([LABEL_INDICES[label]])


#--- model ---

class FFNN(nn.Module):
  # Feel free to add whichever arguments you like here.
  # Note that pretrained_embeds is a numpy matrix of shape (num_embeddings, embedding_dim)

  def __init__(self, pretrained_embeds, n_classes, hidden_layer_1, hidden_layer_2):

      super(FFNN, self).__init__()
      self.pretrained_embeds = pretrained_embeds
      self.input_size = pretrained_embeds.shape[1] * 50 # shape 300 * 50 - input size
      self.hidden_layer_1 = hidden_layer_1 # 256
      self.hidden_layer_2 = hidden_layer_2 # 128
      self.dropout_p = 0.1
      self.n_classes = n_classes # 3: positive, neutral or negative - output size

      # linear layer (15000 x 256)
      self.fc1 = nn.Linear(self.input_size, self.hidden_layer_1)
      # linear layer (256 x 128)
      self.fc2 = nn.Linear(self.hidden_layer_1, self.hidden_layer_2)
      # output layer (128 x 3)
      self.output = nn.Linear(self.hidden_layer_2, self.n_classes)

      # check https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
      self.dropout = nn.Dropout(self.dropout_p)

  def forward(self, x):


      h_1 = F.relu(self.fc1(x))
      h_2 = (F.relu(self.fc2(h_1)))
      output = F.log_softmax(self.output(h_2), dim=0)
      output = output[None, :]
      return output


#--- "main" ---
def get_predictions(output):
    output = torch.squeeze(output)
    pred = torch.max(output)
    if (((output==pred).nonzero().item()) == 0):
        return torch.tensor([0])
    if(((output == pred).nonzero().item()) == 1):
        return torch.tensor([1])
    if (((output == pred).nonzero().item()) == 2):
        return torch.tensor([2])

def get_predictions_name(output):
    pred = torch.max(output)
    if (((output==pred).nonzero().item()) == 0):
        return 'negative'
    if(((output == pred).nonzero().item()) == 1):
        return 'neutral'
    if (((output == pred).nonzero().item()) == 2):
        return 'positive'

def get_tweet_indexes(tweet,  word_to_idx):
    '''
    method to transform each tweet in a torch tensor, containing the embbedings for
    each word
    @input: tweet represented as list of words
    @output: torch tensor representation of tweet
    '''
    # get the indexes of words in each twee t
    indexes = []
    for word in tweet:
        if (word in word_to_idx):
            indexes.append(word_to_idx[word])

    all_tweets.append(indexes)

    return indexes

def make_tweet_embbedings(indexes, pretrained_embeds):

    # each embedding has 300 embeddings for each word
    # the longest tweet in training has 31 words
    # I will create a long np matrix [12.000x1] and append the embbedings of each word in this matrix
    # smaller tweets will have only zeros at the tail
    tweet_emb = []

    for idx in indexes:
        emb = pretrained_embeds[idx]
        tweet_emb.append(emb)

    embeddings = np.concatenate(tweet_emb)
    # add zeros to the end of the array
    zeros_to_be_addeed = 15000 - embeddings.shape[0]
    embeddings = np.pad(embeddings, (0, zeros_to_be_addeed), 'constant')

    tweet_emb = torch.tensor(embeddings, dtype=torch.float32)
    return tweet_emb


if __name__=='__main__':
  #--- data loading ---
  data = read_semeval_datasets(data_dir)
  gensim_embeds = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(model_dir, embeddings_file), binary=True)
  # KeyedVectors<vector_size=300, 14835 keys>
  # mapps a word representation, in this case a index,to a 1D np array
  pretrained_embeds = gensim_embeds.vectors # shape 14835 x 300

  # To convert words in the input tweet to indices of the embeddings matrix:
  word_to_idx = {word: i for i, word in enumerate(gensim_embeds.key_to_index.keys())}
  #--- set up ---
  # WRITE CODE HERE
  model = FFNN(pretrained_embeds, n_classes, hidden_layer_1, hidden_layer_2)
  loss_function = torch.nn.NLLLoss()
  optimizer = optim.SGD(model.parameters(), lr=learning_rate)

  #--- training ---
  for epoch in range(n_epochs):
    print('______________________________________________________________')
    total_loss = 0
    correct_e = 0
    all_tweets = []
    for tweet in data['training']:
      # for each tweet we have: 'ID', 'SENTIMENT', and 'BODY' - body is a list with the words on the tweet

      # get the inputs
      gold_class = label_to_idx(tweet['SENTIMENT']) # this is a tensor

      indexes = get_tweet_indexes(tweet['BODY'], word_to_idx)

      # this is a 1x3 tensor with probs for each class
      # the sum of all values in vector should be 1
      pred = model(make_tweet_embbedings(indexes, pretrained_embeds))

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      # compute the loss based on model output and real labels
      # The input given through a forward call is expected to contain log-probabilities of each class.
      loss = loss_function(pred, gold_class)

      # sum all losses
      total_loss = total_loss + loss.item()

      # backpropagate the loss
      loss.backward()

      # adjust parameters based on the calculated gradients
      optimizer.step()

      correct_e += torch.eq(get_predictions(pred), gold_class)

    if ((epoch+1) % report_every) == 0:
      print(f' epoch: %d, loss: %.4f, accuracy: %.2f' % (epoch,
                                                        total_loss*100/len(data['training']),
                                                        100.0 * correct_e/len(data['training'])
                                                         ))
  # Feel free to use the development data to tune hyperparameters if you like!

  #--- test ---
  correct = 0
  with torch.no_grad():
    for tweet in data['test.gold']:
      # get inputs
      gold_class = label_to_idx(tweet['SENTIMENT'])

      indexes = get_tweet_indexes(tweet['BODY'], word_to_idx)

      # this is a 1x3 tensor with probs for each class
      # the sum of all values in vector should be 1
      pred = model(make_tweet_embbedings(indexes, pretrained_embeds))

      #print('pred  ', get_predictions(pred), 'class ', gold_class)
      correct += torch.eq(get_predictions(pred), gold_class)


      if verbose:
        print('TEST DATA: %s, OUTPUT: %s, GOLD LABEL: %d' % 
              (tweet['BODY'], tweet['SENTIMENT'], get_predictions_name(pred)))
        
    print('test accuracy: %.2f' % (100.0 * correct / len(data['test.gold'])))


