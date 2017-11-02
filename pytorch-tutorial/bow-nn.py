# coding: utf-8

"""
BOW

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
import torch.optim as optim
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


class BOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(BOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.bias = None  ### YOUR CODE HERE ###
        raise NotImplementedError("TODO add parameters")

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        logits = torch.sum(embeds, 1)
        ### YOUR CODE HERE ###
        raise NotImplementedError("TODO add bias")
        return logits


model = BOW(nwords, ntags)
print(model)


def evaluate(model, data):
    """Evaluate a model on a data set."""
    correct = 0.0
    
    for words, tag in data:
        lookup_tensor = Variable(torch.LongTensor([words]))
        scores = model(lookup_tensor)
        predict = scores.data.numpy().argmax(axis=1)[0]

        if predict == tag:
            correct += 1

    return correct, len(data), correct/len(data)


optimizer = optim.SGD(model.parameters(), lr=0.01)

for ITER in range(100):

    random.shuffle(train)
    train_loss = 0.0
    start = time.time()

    for words, tag in train:

        # forward pass
        lookup_tensor = Variable(torch.LongTensor([words]))
        scores = model(lookup_tensor)
        loss = nn.CrossEntropyLoss()
        target = Variable(torch.LongTensor([tag]))
        output = loss(scores, target)
        train_loss += output.data[0]

        # backward pass
        model.zero_grad()
        output.backward()

        # update weights
        optimizer.step()

    print("iter %r: train loss/sent=%.4f, time=%.2fs" % 
          (ITER, train_loss/len(train), time.time()-start))

    # evaluate
    _, _, acc = evaluate(model, dev)
    print("iter %r: test acc=%.4f" % (ITER, acc))

