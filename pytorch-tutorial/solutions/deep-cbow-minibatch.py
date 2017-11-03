# coding: utf-8

"""
Deep CBOW (with minibatching)

Based on Graham Neubig's DyNet code examples:
  https://github.com/neubig/nn4nlp2017-code
  http://phontron.com/class/nn4nlp2017/

"""

from collections import defaultdict
from collections import namedtuple
import time
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
random.seed(1)


CUDA = torch.cuda.is_available()
print("CUDA: %s" % CUDA)


# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]
PAD = w2i["<pad>"]

# One data point
Example = namedtuple("Example", ["words", "tag"])


def read_dataset(filename):
    with open(filename, "r") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            yield Example(words=[w2i[x] for x in words.split(" ")],
                          tag=t2i[tag])


# Read in the data
train = list(read_dataset("data/classes/train.txt"))
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset("data/classes/test.txt"))
nwords = len(w2i)
ntags = len(t2i)


class DeepCBOW(nn.Module):
    """
    Deep CBOW model
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(DeepCBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        h = torch.sum(embeds, 1)
        h = F.relu(self.linear1(h))
        h = F.relu(self.linear2(h))
        h = self.linear3(h)
        return h


model = DeepCBOW(nwords, 16, 16, ntags)

if CUDA:
    model.cuda()

print(model)


def minibatch(data, batch_size=32):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]


def evaluate(model, data):
    """Evaluate a model on a data set."""
    correct = 0.0

    for batch in minibatch(data):

        seqs, tags = preprocess(batch)
        scores = model(get_variable(seqs))
        _, predictions = torch.max(scores.data, 1)
        targets = get_variable(tags)

        correct += torch.eq(predictions, targets).sum().data[0]

    return correct, len(data), correct/len(data)


def get_variable(x):
    """Get a Variable given indices x"""
    tensor = torch.cuda.LongTensor(x) if CUDA else torch.LongTensor(x)
    return Variable(tensor)


def preprocess(batch):
    """ Add zero-padding to a batch. """

    tags = [example.tag for example in batch]

    # add zero-padding to make all sequences equally long
    seqs = [example.words for example in batch]
    max_length = max(map(len, seqs))
    seqs = [seq + [PAD] * (max_length - len(seq)) for seq in seqs]

    return seqs, tags


optimizer = optim.Adam(model.parameters(), lr=0.001)

for ITER in range(100):

    random.shuffle(train)
    train_loss = 0.0
    start = time.time()
    updates = 0

    for batch in minibatch(train):

        updates += 1

        # pad data with zeros
        seqs, tags = preprocess(batch)

        # forward pass
        scores = model(get_variable(seqs))
        targets = get_variable(tags)
        loss = nn.CrossEntropyLoss()
        output = loss(scores, targets)
        train_loss += output.data[0]

        # backward pass
        model.zero_grad()
        output.backward()

        # update weights
        optimizer.step()

    print("iter %r: avg train loss=%.4f, time=%.2fs" %
          (ITER, train_loss/updates, time.time()-start))

    # evaluate
    _, _, acc_train = evaluate(model, train)
    _, _, acc_dev = evaluate(model, dev)
    print("iter %r: train acc=%.4f  test acc=%.4f" % (ITER, acc_train, acc_dev))
