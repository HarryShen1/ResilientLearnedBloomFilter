import numpy as np
import pickle

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

def get_arg_q_thingy(dbf):
	return dbf.B[np.argmax(dbf.W)].n if len(dbf.W) != 0 else -1

with open('bible_corpus.pkl', 'rb') as f:
    bible_corpus = pickle.load(f)
with open('bible_corpus_nouns.pkl', 'rb') as f:
    bible_corpus_nouns = pickle.load(f)
with open('bible_corpus_notnouns.pkl', 'rb') as f:
    bible_corpus_notnouns = pickle.load(f)


def test(lbf_params, bf_params, drlbf_params, FPR_batch_size=1000, num_batch=500, local_scale=1000, scale=2000, dshift_rate=100, n_intervals = 5):
    bf = BloomFilter(*bf_params)
    lbf = LearnedBloomFilter(*lbf_params)
    drlbf = DRLearnedBloomFilter(*drlbf_params)

    bf_fpr = []
    lbf_fpr = []
    drlbf_fpr = []

    def query_true(word):
        if word in bible_corpus_nouns:
            return True
    bf.insert(word)
    drlbf.insert(word)
    lbf.train(word)#help idk wat goes in here
    #dr lbf stuff here

for word in bible_corpus:
    test(word) #idk what params we gonna put here