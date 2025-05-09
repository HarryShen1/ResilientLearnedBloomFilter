import numpy as np
from sklearn.linear_model import LogisticRegression
import hashlib
import random
import matplotlib.pyplot as plt

from LearnedBloomFilter import CountingBloomFilter, ConstantPredictor


class DRLearnedBloomFilter:

	def __init__(self, C, N, k, m, model, preprocess=lambda x : [x], gamma=0.1, X=[]):
		self.C = C 
		self.N = N 
		self.model = model 
		self.preprocess = preprocess

		self.k = k 
		self.m = m

		self.S = []
		self.B = []
		self.W = []

		self.gamma = gamma

		self.phi0 = model()

		self.F = CountingBloomFilter(k, m)
		for x in X:
			self.F.insert(x)

		self.hidden_data = ([], [])

	def insert(self, x):
		preds = [ S[i].predict([self.preprocess(x)])[0] for i in range(len(self.S)) ]
		if True in preds:
			self.B[preds.index(True)].insert(x)
		self.F.insert(x)

	def query(self, x):
		if len(self.S) != 0:
			tau = sum( self.W[i] * self.S[i].predict([self.preprocess(x)])[0] for i in range(len(self.S)) )
			if tau > 0.5:
				return True

			F2 = self.F 
			for b in self.B:
				F2 -= b
			
			return F2.query(x)
		return self.F.query(x)

	def update(self, x, y):
		self.hidden_data[0].append(self.preprocess(x))
		self.hidden_data[1].append(y)

		preds = [phi.predict([self.preprocess(x)])[0] for phi in self.S]

		for i, pred in enumerate(preds):
			if pred != y:
				self.W[i] /= 2

		if y and all([ not b.query(x) for b in self.B ]):
			best = (-1, -1)
			for i, pred in enumerate(preds):
				if pred:
					if self.W[i] > best[1]:
						best = (i, self.W[i])
			if best[0] != -1:
				self.B[best[0]].insert(x)

		if len(self.hidden_data[0]) > self.N:
			if (len(np.unique(self.hidden_data[1])) >= 2):
				self.phi0.fit(*self.hidden_data)
			else:
				self.phi0 = ConstantPredictor(self.hidden_data[1][0])
			if len(self.S) < self.C:
				self.S.append(self.phi0)
				self.B.append(CountingBloomFilter(self.k, self.m))
				self.W.append(1/self.C)
			else:
				i = np.argmin(self.W)

				old_score = self.S[i].score(*self.hidden_data)
				new_score = self.phi0.score(*self.hidden_data)
				
				if type(old_score) == float and type(new_score) == float and type(self.gamma) == float and old_score < new_score - self.gamma:
					print("WHEE")
					self.S[i] = self.phi0
					self.B[i] = CountingBloomFilter(self.k, self.m)
					self.W[i] = 1/self.C

			self.phi0 = self.model()
			self.hidden_data = ([], [])


		c = sum(self.W)
		for i in range(len(self.S)):
			self.W[i] /= c
