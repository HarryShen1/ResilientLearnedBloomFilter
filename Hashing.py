import numpy as np

p = 2 ** 31 - 1

class Hasher: # i hardly know her
    def __init__(self, k, i): # k hashes, i-wise independent
        self.i = i 
        self.hashes = np.vectorize(int)(np.random.rand(k, i) * (p + 1))


    def string_hash(self, string):
        h = 5381
        for c in string:
            h = ((h << 5) + h) + ord(c)
        return h

    def hash(self, value):
        c = value
        if type(value) == str:
            c = self.string_hash(value)
        c = int(c)

        arr = c ** np.arange(self.i)
        return ((self.hashes @ arr) % p).astype(int)

