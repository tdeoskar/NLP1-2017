# Lab 1

Getting acquainted with language data and language models.

## Defaultdict

We've edited the hint on using the defaultdict for smoothing. This
```python
ngram = defaultdict(lambda: k/(N + kV), ngram)
```
has been changed into
```python
d = bigram_frequencies[history] = {w1: 120, w2: 340, w3: 7 ...} # your bigram counts for 'history'
N = sum(d.values()) # sum of actual counts for 'history'
d_normalized = .... # normalize d with smoothing applied to elements already in the dictionary  
d_smoothed = defaultdict(lambda: k/(N + kV), d_normalized) # add smoothing to 'unseen' words
```
to avoid confusing you. See [piazza](https://piazza.com/class/) for a longer explanation.

## Error

There is a small error in the original version for the function `read` (exercise 2.1). The new version has been updated here on github.

You can also copy-paste the code below if you have already started on the notebook:

```python
def train_ngram(data, N, k=0):
    """
    Trains an n-gram language model with optional add-k smoothing
    and additionaly returns the unigram model

    :param data: text-data as returned by read
    :param N: (N>1) the order of the ngram e.g. N=2 gives a bigram
    :param k: optional add-k smoothing
    :returns: ngram and unigram
    """
    ngram = defaultdict(Counter) # ngram[history][word] = #(history,word)
    unpacked_data = [word for sent in data for word in sent]
    unigram = defaultdict(float, Counter(unpacked_data)) # default prob is 0.0           

    ## YOUR CODE HERE ##

    return ngram, unigram
```
