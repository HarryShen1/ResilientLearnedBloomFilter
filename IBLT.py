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
        hash_bytes = hashlib.sha256(str(x).encode()).digest()
        seeds = [int.from_bytes(hash_bytes[i:i+4], 'little') for i in range(0, self.k * 4, 4)]
        return [seed % self.m for seed in seeds]
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
                decoded.add((cell.key_sum,cell.value_sum))
                self.delete(cell.key_sum)
        return decoded
    def hashValue(self, x):
        return int(hashlib.md5(str(x).encode()).hexdigest(), 16)
def test_iblt():
    iblt = IBLT(m=20, k=4)

    # Insert elements
    elements_to_insert = [10, 20, 30, 40, 50]
    for x in elements_to_insert:
        iblt.insert(x)
        print(f"Inserted: {x}")

    # Attempt to get inserted elements
    print("\nChecking inserted elements:")
    for x in elements_to_insert:
        result = iblt.get(x)
        print(f"Get({x}) -> {result}")

    # Delete some elements
    elements_to_delete = [20, 30]
    for x in elements_to_delete:
        iblt.delete(x)
        print(f"Deleted: {x}")

    # Try to get deleted elements
    print("\nChecking deleted elements:")
    for x in elements_to_delete:
        result = iblt.get(x)
        print(f"Get({x}) -> {result}")

    # Try to list entries remaining in the table
    print("\nListing entries:")
    entries = iblt.listEntries()
    for key, hashed_value in entries:
        print(f"Key: {key}, Hashed Value: {hashed_value}")

# Run the test
test_iblt()