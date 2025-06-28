p = 2 ** 61 - 1

class Hasher: # i hardly know her
    def __init__(k, i): # k hashes, i-wise independent
        self.i = i 
        self.hashes = np.vectorize(int)(np.random.rand(k, i) * (p + 1))


    def string_hash(string):
        hash = 5381

        for c in string:
            hash = ((hash << 5) + hash) + ord(c)

        return hash

    def hash(value):
        c = value
        if type(value) == str:
            c = string_hash(value)

        arr = x ** np.arange(self.i)
        return ((self.hashes @ arr) % p)

