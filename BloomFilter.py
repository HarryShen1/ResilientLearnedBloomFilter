import numpy as np

k = 10
p = 261 - 1
h = np.vectorize(int)(np.random.rand(k, k) * (p+1))
m = 1000

def hash(x):
    arr = x ** np.arange(k)
    return ((h @ arr) % p) % m

#create n by k array of some numbers to make table that is bloom filter
#to add set all bits to 1
z = np.zeros((k, m))

def insert(z, x):
  z[np.arange(k), hash(x)] = 1

def query(z, x):
  return all(z[np.arange(k), hash(x)])

insert(z,7)
print(query(z,7))
print(query(z,5))