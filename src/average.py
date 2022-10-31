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
# from allennlp.nn.util import get_text_field_mask

# Add the path to these data manipulation files if necessary:
# import sys
# sys.path.append('</PATH/TO/DATA/MANIP/FILES>')
from data_semeval import *
from paths import data_dir, model_dir

# name of the embeddings file to use
# Alternatively, you can also use the text file GoogleNews-pruned2tweets.txt (from Moodle),
# or the full set, wiz. GoogleNews-vectors-negative300.bin (from https://code.google.com/archive/p/word2vec/)
embeddings_file = 'GoogleNews-pruned2tweets.bin'

# --- hyperparameters ---

# Feel free to experiment with different hyperparameters to see how they compare!
# You can turn in your assignment with the best settings you find.

n_classes = len(LABEL_INDICES)
n_epochs = 30
learning_rate = 0.0001
hidden_layer_1 = 50
report_every = 1
verbose = False


# --- auxilary functions ---

# To convert string label to pytorch format:
def label_to_idx(label):
    return torch.LongTensor([LABEL_INDICES[label]])


# --- model ---

class FFNN(nn.Module):
    # Feel free to add whichever arguments you like here.
    # Note that pretrained_embeds is a numpy matrix of shape (num_embeddings, embedding_dim)

    def __init__(self, pretrained_embeds, n_classes, hidden_layer_1, extra_arg_2=None):
        super(FFNN, self).__init__()
        self.pretrained_embeds = pretrained_embeds
        self.input_size = pretrained_embeds.shape[1]  # shape 14835 x 300 - input size
        self.hidden_layer_1 = hidden_layer_1  # size will be defined
        self.dropout_p = 0.1
        self.n_classes = n_classes  # 3: positive, neutral or negative - output size

        # linear layer (300 x 100)
        self.fc1 = nn.Linear(self.input_size, self.hidden_layer_1)
        # linear layer (100 x 3 - n_classes)
        self.fc2 = nn.Linear(self.hidden_layer_1, self.n_classes)
        # check https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, x):
        h_1 = self.fc1(x)
        h_1 = self.dropout(F.relu(h_1))
        output = F.log_softmax(self.fc2(h_1), dtype=torch.float)
        output = output[None, :]
        return output


# --- "main" ---
def get_predictions(output):
    # print('output', output)
    output = torch.squeeze(output)
    pred = torch.max(output)
    # print(pred)
    if (((output == pred).nonzero().item()) == 0):
        # print(torch.tensor([0]))
        # print('0')
        return torch.tensor([0])
    if (((output == pred).nonzero().item()) == 1):
        # print(torch.tensor([1]))
        # print('1')
        return torch.tensor([1])
    if (((output == pred).nonzero().item()) == 2):
        # print(torch.tensor([2]))
        # print('2')
        return torch.tensor([2])


def get_predictions_name(output):
    pred = torch.max(output)
    if (((output == pred).nonzero().item()) == 0):
        return 'negative'
    if (((output == pred).nonzero().item()) == 1):
        return 'neutral'
    if (((output == pred).nonzero().item()) == 2):
        return 'positive'


def get_target_tensor(label):
    # each element in target has to have 0 <= value < C
    # c = n. classes = 3
    if (label == 'negative'):
        return torch.tensor([1, 0, 0], dtype=torch.float)
    if (label == 'neutral'):
        return torch.tensor([0, 1, 0], dtype=torch.float)
    if (label == 'positive'):
        return torch.tensor([0, 0, 1], dtype=torch.float)


def get_tweet_indexes(tweet, word_to_idx):
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


def word2vex_average(indexes, pretrained_embeds):
    # Initialize word2vec_feature vector
    # adds the word2vec average
    average_vec = np.zeros(300)
    for idx in indexes:
        if pretrained_embeds[idx] in pretrained_embeds:
            average_vec += (pretrained_embeds[idx] / len(indexes))
        else:
            pass
    av = torch.tensor(average_vec, dtype=torch.float32).squeeze()
    #print(av)
    #print(av.shape)
    return av


if __name__ == '__main__':
    # --- data loading ---
    data = read_semeval_datasets(data_dir)
    gensim_embeds = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(model_dir, embeddings_file),
                                                                    binary=True)
    # KeyedVectors<vector_size=300, 14835 keys>
    # mapps a word representation, in this case a index,to a 1D np array
    pretrained_embeds = gensim_embeds.vectors  # shape 14835 x 300

    # To convert words in the input tweet to indices of the embeddings matrix:
    word_to_idx = {word: i for i, word in enumerate(gensim_embeds.key_to_index.keys())}
    # --- set up ---
    # WRITE CODE HERE
    model = FFNN(pretrained_embeds, n_classes, hidden_layer_1)
    loss_function = torch.nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # --- training ---
    for epoch in range(n_epochs):
        print('______________________________________________________________')
        total_loss = 0
        correct_e = 0
        all_tweets = []
        for tweet in data['training']:
            # for each tweet we have: 'ID', 'SENTIMENT', and 'BODY' - body is a list with the words on the tweet

            # get the inputs
            gold_class = label_to_idx(tweet['SENTIMENT'])  # this is a tensor
            target = get_target_tensor(tweet['SENTIMENT'])  # one hot enconded representation of labels

            indexes = get_tweet_indexes(tweet['BODY'], word_to_idx)
            # .append(indexes)

            # this is a 1x3 tensor with probs for each class
            # the sum of all values in vector should be 1
            pred = model(word2vex_average(indexes, pretrained_embeds))


            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # compute the loss based on model output and real labels
            # The input given through a forward call is expected to contain log-probabilities of each class.
            #print(pred, target, get_predictions(pred), gold_class)
            loss = loss_function(pred, gold_class)

            # sum all losses
            total_loss = total_loss + loss.item()

            # backpropagate the loss
            loss.backward()

            # adjust parameters based on the calculated gradients
            optimizer.step()

            correct_e += torch.eq(get_predictions(pred), gold_class)
            #print('pred ', get_predictions(pred), 'class ', gold_class)
            # print(correct_e)

        # print statistics
        if ((epoch + 1) % report_every) == 0:
            print('epoch: %d, loss: %.4f' % (epoch, total_loss * 100 / len(data['training'])))
            print('train accuracy: %.2f' % (100.0 * correct_e / len(data['training'])))
    # Feel free to use the development data to tune hyperparameters if you like!

    # --- test ---
    correct = 0
    with torch.no_grad():
        for tweet in data['test.gold']:
            # get inputs
            gold_class = label_to_idx(tweet['SENTIMENT'])

            indexes = get_tweet_indexes(tweet['BODY'], word_to_idx)
            # .append(indexes)

            # this is a 1x3 tensor with probs for each class
            # the sum of all values in vector should be 1
            pred = model(word2vex_average(indexes, pretrained_embeds))

            # print('pred  ', get_predictions(pred), 'class ', gold_class)
            correct += torch.eq(get_predictions(pred), gold_class)

            if verbose:
                print('TEST DATA: %s, OUTPUT: %s, GOLD LABEL: %d' %
                      (tweet['BODY'], tweet['SENTIMENT'], get_predictions_name(pred)))

        print('test accuracy: %.2f' % (100.0 * correct / len(data['test.gold'])))


