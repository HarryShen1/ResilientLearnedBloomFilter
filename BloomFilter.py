import numpy as np
from Hashing import Hasher

k = 10
h = np.vectorize(int)(np.random.rand(k, k) * (p + 1))
m = 1000

class BloomFilter:
    def __init__(self,k,p,h,m):
        self.k=k
        self.p=p
        self.hasher = Hasher(k, k)
        self.m=m
        self.data = np.zeros((k, m))

    def insert(self,x):
      self.data[np.arange(k), self.hasher.hash(x) % self.m] = 1

    def query(self,x):
      return all(self.data[np.arange(k), self.hasher.hash(x) % self.m])

bloomfilter = BloomFilter(k,p,h,m)
bloomfilter.insert("hi")
print(bloomfilter.query("hi"))
print(bloomfilter.query("no"))