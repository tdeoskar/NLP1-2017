# coding: utf-8

"""
BOW (simple version)

Based on Graham Neubig's DyNet code examples:
  https://github.com/neubig/nn4nlp2017-code
  http://phontron.com/class/nn4nlp2017/

"""

from collections import defaultdict
import time
import random
import torch
from torch.autograd import Variable
import torch.nn as nn

torch.manual_seed(42)


# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]


def read_dataset(filename):
    with open(filename, "r") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            yield ([w2i[x] for x in words.split(" ")], t2i[tag])


# Read in the data
train = list(read_dataset("data/classes/train.txt"))
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset("data/classes/test.txt"))
nwords = len(w2i)
ntags = len(t2i)


# The parameters for our BoW-model
dtype = torch.FloatTensor  # enable CUDA here if you like
w = Variable(torch.randn(nwords, ntags).type(dtype), requires_grad=True)
### YOUR CODE HERE ###
raise NotImplementedError("TODO add bias parameters")


# A function to calculate scores for one sentence
def calc_scores(words):
    lookup_tensor = Variable(torch.LongTensor(words))
    embed = w[lookup_tensor]
    score = torch.sum(embed, 0)
    ### YOUR CODE HERE ###
    raise NotImplementedError("TODO add bias")
    return score.view((1, -1))


for ITER in range(100):
    
    # train
    random.shuffle(train)
    train_loss = 0.0
    start = time.time()
    
    for words, tag in train:
        
        # forward pass
        scores = calc_scores(words)
        target = Variable(torch.LongTensor([tag]))        
        loss = nn.CrossEntropyLoss()
        output = loss(scores, target)
        train_loss += output.data[0]        
        
        # backward pass (compute gradients)
        output.backward()

        # update weights with SGD
        lr = 0.01
        w.data -= lr * w.grad.data
        ### YOUR CODE HERE ###
        raise NotImplementedError("TODO update bias parameters")

        # clear gradients for next step
        w.grad.data.zero_()
        ### YOUR CODE HERE ###
        raise NotImplementedError("TODO zero out bias gradients")

        
    print("iter %r: train loss/sent=%.4f, time=%.2fs" % 
          (ITER, train_loss/len(train), time.time()-start))

    # evaluate
    correct = 0.0
    for words, tag in dev:
        scores = calc_scores(words)
        predict = scores.data.numpy().argmax(axis=1)
        if predict == tag:
            correct += 1
    
    print("iter %r: test acc=%.4f" % 
          (ITER, correct/len(dev)))
