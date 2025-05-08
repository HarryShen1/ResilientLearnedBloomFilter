import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestCentroid
import hashlib
import random
import matplotlib.pyplot as plt
from tqdm import tqdm 
import sys

from LearnedBloomFilter import BloomFilter, CountingBloomFilter, LearnedBloomFilter
from DRLearnedBloomFilter import DRLearnedBloomFilter

# fixing memory !!! 
memory = 5000
n_hash = 5

bf_params = (n_hash, memory // n_hash) # (k, m)

# Memory usage = k * m

lbf_params = (n_hash, memory // n_hash, LogisticRegression, lambda x : [x, x**2]) # (k, m, confidence_threshold)

# Memory usage = k * m + 1 ~~ km

drlbf_params = (2, 50, n_hash, memory // (3 * n_hash), LogisticRegression, lambda x : [x, x**2]) # (C, N, k, m, model)

# Memory usage = (k * m) * (C + 1) + C * 1 ~~ (C + 1)(km + 1) ~~ (C + 1)km
# 3km in this case

def test(lbf_params, bf_params, drlbf_params, FPR_batch_size=200, num_batch=100, local_scale=10, scale=100, dshift_rate=25):
	bf = BloomFilter(*bf_params)
	lbf = LearnedBloomFilter(*lbf_params)
	drlbf = DRLearnedBloomFilter(*drlbf_params)

	bf_fpr = []
	lbf_fpr = []
	drlbf_fpr = []

	# Random interval with noise
	a, b = sorted(np.random.randn(2) * scale)

	for i in range(int(a), int(b)):
		bf.insert(i)
		drlbf.insert(i)

	train_universe = np.random.randn(1000) * scale
	lbf.train(np.arange(int(a), int(b)), train_universe[(train_universe < a) | (train_universe > b)])

	Q = 0

	# print(f"Interval is from {a} to {b}")
	
	for i in tqdm(range(num_batch), leave=False):
		# Random query distribution
		if (i % dshift_rate == 0) and (i != 0):
			Q = (np.random.randn(1) * scale)[0]

		queries = (np.random.randn(FPR_batch_size) * local_scale + Q).astype(int)

		def evaluate(method, x):
			return [method.query(i) != (a < i and i < b) for i in x]

		bf_fpr.append(sum(evaluate(bf, queries))/FPR_batch_size)
		lbf_fpr.append(sum(evaluate(lbf, queries))/FPR_batch_size)

		def evaluate_dr(method, x):
			return [(method.query(i), method.update(i, (a < i and i < b)))[0] != (a < i and i < b) for i in x]

		drlbf_fpr.append(sum(evaluate_dr(drlbf, queries))/FPR_batch_size)

		# print(f"Batch done, results: {bf_fpr[-1]}, {lbf_fpr[-1]}, {drlbf_fpr[-1]}.")

	# print("Total done. Plotting.")

	# print("BF: ", bf_fpr)
	# print("LBF: ", lbf_fpr)
	# print("DRLBF: ", drlbf_fpr)

	return np.array([bf_fpr, lbf_fpr, drlbf_fpr])


n_b = 100

X = test(lbf_params, bf_params, drlbf_params, num_batch = n_b)
for i in tqdm(range(2, 51)):
	X = (1 - 1/i) * X + (1/i) * test(lbf_params, bf_params, drlbf_params, num_batch = n_b)

plt.plot(np.arange(n_b), X[0], label="BF")
plt.plot(np.arange(n_b), X[1], label="LBF")
plt.plot(np.arange(n_b), X[2], label="DRLBF")

plt.legend()

plt.show()