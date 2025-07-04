import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import GradientBoostingClassifier
import hashlib
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

from LearnedBloomFilter import BloomFilter, CountingBloomFilter, LearnedBloomFilter
from DRLearnedBloomFilter import DRLearnedBloomFilter

from Word2VecPreprocessor import preprocess


# fixing memory !!!
memory = 10000
n_hash = 5

scale = 2000

bf_params = (n_hash, memory // n_hash) # (k, m)

# Memory usage = k * m

lbf_params = (n_hash, memory // n_hash, lambda : GradientBoostingClassifier(n_estimators = 10), preprocess) # (k, m, confidence_threshold)

# Memory usage = k * m + 1 ~~ km

drlbf_params = (2, 1000, n_hash, memory // (3 * n_hash), lambda : GradientBoostingClassifier(n_estimators = 10), preprocess) # (C, N, k, m, model)

# Memory usage = (k * m) * (C + 1) + C * 1 ~~ (C + 1)(km + 1) ~~ (C + 1)km
# 3km in this case

with open('bible_corpus.pkl', 'rb') as f:
    bible_corpus = pickle.load(f)
with open('bible_corpus_nouns.pkl', 'rb') as f:
    bible_corpus_nouns = pickle.load(f)
with open('bible_corpus_notnouns.pkl', 'rb') as f:
    bible_corpus_notnouns = pickle.load(f)

def lbf_train(lbf_params,size):
    lbf = LearnedBloomFilter(*lbf_params)
    universe = bible_corpus
    positive_samples = []
    positive_counter = 0
    negative_samples = []
    negative_counter = 0
    bible_corpus_nouns_set = set(bible_corpus_nouns)
    for i in universe:
        if i in bible_corpus_nouns_set:
            positive_samples.append(i)
        else:
            negative_samples.append(i)
    lbf.train(universe, positive_samples, negative_samples)
    return (lbf,positive_counter)

def test(bf, lbf, drlbf, fpr, word):

    bf_fpr = fpr[0]
    lbf_fpr = fpr[1]
    drlbf_fpr = fpr[2]

    def query_true():
        if word in bible_corpus_nouns:
            return True
        return False

    def evaluate(method):
        if query_true()==method.query(word):
            return True
        return False

    def evaluate_dr(method):
        if query_true()==method.query(word):
            method.update(word, query_true())
            return True
        method.update(word, query_true())
        return False

    bf_fpr.append(evaluate(bf))
    lbf_fpr.append(evaluate(lbf))
    drlbf_fpr.append(evaluate(drlbf))
    return [bf_fpr, lbf_fpr, drlbf_fpr]

size = 1000 # how many words you want to train the bf and lbf on

bf = BloomFilter(*bf_params)
train = lbf_train(lbf_params, size)
lbf = train[0]
positive_counter = train[1]
drlbf =  DRLearnedBloomFilter(*drlbf_params)

for i in tqdm(range(len(bible_corpus_nouns))):
    word = bible_corpus_nouns[i]
    bf.insert(word)
    drlbf.insert(word)
    # if i >= positive_counter:
    #     lbf.insert(word)

bf_fpr=[]
lbf_fpr=[]
drlbf_fpr=[]
fpr = [bf_fpr,lbf_fpr,drlbf_fpr]


fprs = [ [], [], [] ]

plt.ion()

COLORS = ['blue', 'green', 'red']
LABELS = ['Bloom Filter', 'Learned Bloom Filter', 'Distribution-Resilient Filter']

AVG_INTERVAL = 1000

for word in tqdm(bible_corpus):
    fpr = test(bf, lbf, drlbf, fpr, word)
    
    n = len(fpr[0])


    if n != 0 and n % AVG_INTERVAL == 0:
        plt.cla()

        for i in range(3):
            fprs[i].append(sum(fpr[i][-AVG_INTERVAL:]))

            plt.plot(np.arange(n // AVG_INTERVAL), fprs[i], c=COLORS[i], label=LABELS[i])
            plt.legend()

        plt.draw()
        plt.pause(0.001)
