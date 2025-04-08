import numpy as np

k = 10
p = 2 ** 61 - 1
h = np.vectorize(int)(np.random.rand(k, k) * (p + 1))
m = 1000

class BloomFilter:
    def __init__(self,k,p,h,m):
        self.k=k
        self.p=p
        self.h=h
        self.m=m
        self.data = np.zeros((k, m))

    def hash(self,x):
        arr = x ** np.arange(k)
        return ((self.h @ arr) % self.p) % self.m

#create n by k array of some numbers to make table that is bloom filter
#to add set all bits to 1

    def insert(self,x):
      self.data[np.arange(k), hash(x)] = 1

    def query(self,x):
      return all(self.data[np.arange(k), hash(x)])

bloomfilter = BloomFilter(k,p,h,m)
bloomfilter.insert(7)
print(bloomfilter.query(7))
print(bloomfilter.query(5))