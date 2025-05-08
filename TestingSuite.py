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

lbf_params = (5, 1000, LogisticRegression) # (k, m, confidence_threshold)
bf_params = (5, 1000) # (k, m)
drlbf_params = (2, 50, 5, 400, LogisticRegression) # (C, N, k, m, model)

def test(lbf_params, bf_params, drlbf_params, FPR_batch_size=100, num_batch=50, local_scale=10, scale=100, dshift_rate=10):
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

	print(f"Interval is from {a} to {b}")
	
	for i in tqdm(range(num_batch)):
		# Random query distribution
		if (i % dshift_rate == 0) and (i > 0):
			Q = (np.random.randn(1) * scale)[0]

		queries = (np.random.randn(FPR_batch_size) * local_scale + Q).astype(int)

		def evaluate(method, x):
			return [method.query(i) != (a < i and i < b) for i in x]

		bf_fpr.append(sum(evaluate(bf, queries))/FPR_batch_size)
		lbf_fpr.append(sum(evaluate(lbf, queries))/FPR_batch_size)

		def evaluate_dr(method, x):
			return [(method.query(i), method.update(i, (a < i and i < b)))[0] != (a < i and i < b) for i in x]

		drlbf_fpr.append(sum(evaluate_dr(drlbf, queries))/FPR_batch_size)

		print(f"Batch done, results: {bf_fpr[-1]}, {lbf_fpr[-1]}, {drlbf_fpr[-1]}.")

	print("Total done. Plotting.")

	print("BF: ", bf_fpr)
	print("LBF: ", lbf_fpr)
	print("DRLBF: ", drlbf_fpr)

	plt.plot(np.arange(num_batch), bf_fpr, label="BF")
	plt.plot(np.arange(num_batch), lbf_fpr, label="LBF")
	plt.plot(np.arange(num_batch), drlbf_fpr, label="DRLBF")

	plt.legend()

	plt.show()



test(lbf_params, bf_params, drlbf_params)