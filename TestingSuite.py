import numpy as np
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

def get_arg_q_thingy(dbf):
	return dbf.B[np.argmax(dbf.W)].n if len(dbf.W) != 0 else -1

def test(lbf_params, bf_params, drlbf_params, FPR_batch_size=1000, num_batch=500, local_scale=1000, scale=2000, dshift_rate=100, n_intervals = 5):
	bf = BloomFilter(*bf_params)
	lbf = LearnedBloomFilter(*lbf_params)
	drlbf = DRLearnedBloomFilter(*drlbf_params)

	bf_fpr = []
	lbf_fpr = []
	drlbf_fpr = []

	# Random interval with noise
	INTERVALS = sorted(np.random.randn(n_intervals * 2) * scale)


	def get_all_points():
		L = []
		for i in range(n_intervals):
			L += list(np.arange(int(INTERVALS[2*i])+1, int(INTERVALS[2*i+1])))
		return L

	# print(len(get_all_points()))


	def query_true(x):
		return any([ INTERVALS[2*i] < x < INTERVALS[2*i+1] for i in range(n_intervals) ])

	for i in get_all_points():
		bf.insert(i)
		drlbf.insert(i)


	Q = 0

	train_universe = np.random.randn(1000) * local_scale
	lbf.train(get_all_points(), train_universe[np.vectorize(query_true)(train_universe)], train_universe[np.vectorize(lambda x : not query_true(x))(train_universe)])
	
	# plt.ion()
	for i in tqdm(range(num_batch), leave=False):
		# print(get_arg_q_thingy(drlbf))
		# Random query distribution
		if (i % dshift_rate == 0) and (i != 0):
			Q = (np.random.randn(1) * scale)[0]
			queries = (np.random.randn(FPR_batch_size) * local_scale + Q).astype(int)
			# print(np.average((queries > a) & (queries < b)))

		queries = (np.random.randn(FPR_batch_size) * local_scale + Q).astype(int)

		def evaluate(method, x):
			return [method.query(i) != query_true(i) for i in x]

		bf_fpr.append(sum(evaluate(bf, queries))/FPR_batch_size)
		
		def evaluate_l(method, x):
			arr_a = [method.query(i) for i in x]
			arr_b = [query_true(i) for i in x] # we have the meats
			# plt.clf()
			# plt.scatter(x, arr_a, c=arr_b)
			# plt.pause(0.1)
			return [arr_a[i] != arr_b[i] for i in range(len(x))]

		lbf_fpr.append(sum(evaluate_l(lbf, queries))/FPR_batch_size)

		def evaluate_dr(method, x):
			arr_a = [(method.query(i), method.update(i, query_true(i)))[0] for i in x]
			arr_b = [query_true(i) for i in x] # we have the meats
			# plt.clf()
			# plt.scatter(x, arr_a, c=arr_b)
			# plt.pause(0.1)
			return  [arr_a[i] != arr_b[i] for i in range(len(x))]

		drlbf_fpr.append(sum(evaluate_dr(drlbf, queries))/FPR_batch_size)

		# print(f"Batch done, results: {bf_fpr[-1]}, {lbf_fpr[-1]}, {drlbf_fpr[-1]}.")
	# plt.show()
	# plt.clf()
	# plt.ioff()

	# print("Total done. Plotting.")

	# print("BF: ", bf_fpr)
	# print("LBF: ", lbf_fpr)
	# print("DRLBF: ", drlbf_fpr)

	return np.array([bf_fpr, lbf_fpr, drlbf_fpr])

## TODO harry make 500

n_b = 500

## TODO harry make 50
n_iter = 50

X = test(lbf_params, bf_params, drlbf_params, num_batch = n_b)
for i in tqdm(range(2, n_iter + 1)):
	X = (1 - 1/i) * X + (1/i) * test(lbf_params, bf_params, drlbf_params, num_batch = n_b)



plt.xlabel("Iteration (1000 queries)")
plt.ylabel("False Positive Rate (FPR)")

plt.title("FPR against Time with Distribution Shift")

plt.plot(np.arange(n_b), X[0], label="BF")
plt.plot(np.arange(n_b), X[1], label="LBF")
plt.plot(np.arange(n_b), X[2], label="DRLBF")

plt.legend()

plt.savefig("out_thingy.png")
plt.show()