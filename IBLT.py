import random
import numpy as np
import hashlib

class IBLTCell:
    def __init__(self):
        self.count = 0
        self.key_sum = 0
        self.value_sum = 0
class IBLT:
    def __init__(self, m, k):
        self.m = m  # number of cells
        self.k = k  # number of hash functions
        self.table = [IBLTCell() for _ in range(m)]
    def hashes(self, x):
        random.seed(hash(x))
        return [random.randint(0, self.m-1) for _ in range(self.k)]
    def insert(self, x):
        for h in self.hashes(x):
            self.table[h].count+=1
            self.table[h].key_sum+=x
            self.table[h].value_sum+=self.hashValue(x)
    def delete(self, x):
        for h in self.hashes(x):
            self.table[h].count-=1
            self.table[h].key_sum-=x
            self.table[h].value_sum-=self.hashValue(x)
    def get(self, x):
        for h in self.hashes(x):
            if self.table[h].count==0:
                return None
            elif self.table[h].count == 1:
                if self.table[h].key_sum == x:
                    return self.table[h].value_sum
                else:
                    return None
        return "Not found"
    def listEntries(self):
        decoded = set()
        for cell in self.table:
            if cell.count ==1:
                decoded.add((self.table[cell].key_sum,self.table[cell].value_sum))
                self.delete(cell.key_sum)
    def hashValue(self, x):
        return hash(x)
