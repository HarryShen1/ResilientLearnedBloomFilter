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

# fixing memory !!!
memory = 10000
n_hash = 5

scale = 2000

bf_params = (n_hash, memory // n_hash) # (k, m)

# Memory usage = k * m

lbf_params = (n_hash, memory // n_hash, lambda : GradientBoostingClassifier(n_estimators = 10), lambda x : [x/scale]) # (k, m, confidence_threshold)

# Memory usage = k * m + 1 ~~ km

drlbf_params = (2, 1000, n_hash, memory // (3 * n_hash), lambda : GradientBoostingClassifier(n_estimators = 10), lambda x : [x/scale]) # (C, N, k, m, model)

# Memory usage = (k * m) * (C + 1) + C * 1 ~~ (C + 1)(km + 1) ~~ (C + 1)km
# 3km in this case

with open('bible_corpus.pkl', 'rb') as f:
    bible_corpus = pickle.load(f)
with open('bible_corpus_nouns.pkl', 'rb') as f:
    bible_corpus_nouns = pickle.load(f)
with open('bible_corpus_notnouns.pkl', 'rb') as f:
    bible_corpus_notnouns = pickle.load(f)

def bfTrain(bf_params,size):
    bf = BloomFilter(*bf_params)
    universe = bible_corpus[:size]
    positive_samples = []
    positive_counter = 0
    for i in universe:
        if i == bible_corpus_nouns[positive_counter]:
            positive_samples.append(i)
            positive_counter += 1
    for i in positive_samples:
        bf.insert(i)
    return bf

def lbfTrain(lbf_params,size):
    lbf = LearnedBloomFilter(*lbf_params)
    universe = bible_corpus[:size]
    positive_samples = []
    positive_counter = 0
    negative_samples = []
    negative_counter = 0
    for i in universe:
        if i == bible_corpus_nouns[positive_counter]:
            positive_samples.append(i)
            positive_counter += 1
        else:
            negative_samples.append(i)
            negative_counter += 1
    lbf.train(universe, positive_samples, negative_samples)
    return lbf

def test(bf, lbf, drlbf_params, fpr, word):
    drlbf = DRLearnedBloomFilter(*drlbf_params)

    bf_fpr = fpr[0]
    lbf_fpr = lbf[1]
    drlbf_fpr = lbf[2]

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

    bf_fpr.append(evaluate(bf))#HELPHELPHELP
    lbf_fpr.append(evaluate(lbf))#HELPHELPHELP
    drlbf_fpr.append(evaluate(drlbf))#HELPHELPEHELP
    return np.array([bf_fpr, lbf_fpr, drlbf_fpr])

size = 1000 # how many words you want to train the bf and lbf on

bf = bfTrain(bf_params, size)
lbf = lbfTrain(lbf_params, size)

bf_fpr=[]
lbf_fpr=[]
drlbf_fpr=[]
fpr = [bf_fpr,lbf_fpr,drlbf_fpr]

for word in bible_corpus:
    fpr = test(bf, lbf, drlbf_params, fpr, word)

