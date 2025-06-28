import numpy as np
from sklearn.linear_model import LogisticRegression
import hashlib
import matplotlib.pyplot as plt
from Hashing import Hasher


class ConstantPredictor:
    def __init__(self, x):
        self.x = x  

    def predict(self, y):
        return [self.x]

    def score(self, X, Y):
        return [self.x == y for y in Y]

# Simplified, working BloomFilter with deterministic hashing
class BloomFilter:
    def __init__(self, k, m):
        self.k = k
        self.m = m
        self.hasher = Hasher(k, k)
        self.data = np.zeros((k, m))

    def insert(self, x):
        indices = self.hasher.hash(x) % self.m
        self.data[np.arange(self.k), indices] = 1

    def query(self, x):
        indices = self.hasher.hash(x) % self.m
        return all(self.data[np.arange(self.k), indices])


class CountingBloomFilter:
    def __init__(self, k, m, data=None):
        self.k = k
        self.m = m
        self.hasher = Hasher(k, k)
        if data is None:
            self.data = np.zeros((k, m))
        else:
            self.data = data

        self.n = 0

    def insert(self, x):
        self.n += 1
        indices = self.hasher.hash(x) % self.m
        self.data[np.arange(self.k), indices] += 1

    def query(self, x):
        indices = self.hasher.hash(x) % self.m
        return all(self.data[np.arange(self.k), indices])

    def __add__(self, other):
        return CountingBloomFilter(self.k, self.m, self.data + other.data)

    def __neg__(self, other):
        return CountingBloomFilter(self.k, self.m, -self.data)

    def __sub__(self, other):
        return CountingBloomFilter(self.k, self.m, self.data - other.data)



# Learned Bloom Filter
class LearnedBloomFilter:
    def __init__(self, k, m, model, preprocess=lambda x : [x]):
        self.k = k
        self.m = m
        self.backup_filter = BloomFilter(k, m)
        self.preprocess = preprocess
        self.model = model()
        self.trained = False

    # def _get_features(self, x):
    #     indices = self.backup_filter.hash(x)
    #     features = np.zeros(self.m)
    #     features[indices] = 1
    #     return features

    def train(self, X, positive_samples, negative_samples):
        X_train = []
        y_train = []

        for x in positive_samples:
            X_train.append(self.preprocess(x))
            y_train.append(1)

        for x in negative_samples:
            X_train.append(self.preprocess(x))
            y_train.append(0)

        if (len(np.unique(y_train)) > 1):
            self.model.fit(X_train, y_train)
        else:
            self.model = ConstantPredictor(y_train[0])

        # plt.scatter(X_train, y_train, c=self.model.predict(X_train))
        # plt.show()

        self.trained = True

        # Selectively add uncertain positives to backup
        for x in X:
            prob = self.model.predict([self.preprocess(x)])[0]
            if not prob:
                self.backup_filter.insert(x)

    def insert(self, x):
        self.backup_filter.insert(x)

    def query(self, x):
        if not self.trained:
            raise RuntimeError("Model not trained.")
        prob = self.model.predict([self.preprocess(x)])[0]
        if prob:
            return True
        else:
            return self.backup_filter.query(x)


if __name__ == "__main__":
    # Parameters
    k = 20          # Number of hash functions
    m = 2000        # Bloom filter size
    threshold = 0.9
    n = 10000       # Number of Gaussian samples

    # Setup
    backup = BloomFilter(k, m)
    lbf = LearnedBloomFilter(k, m, threshold, backup)

    # Generate Gaussian data
    samples = np.random.normal(loc=0, scale=1, size=n)
    positive_samples = samples[samples > 1]
    negative_samples = samples[samples <= 1]

    print(f"Training on {len(positive_samples)} positives and {len(negative_samples)} negatives...")

    # Train the learned Bloom filter
    lbf.train(positive_samples, negative_samples)
    print("Model coefficient norm:", np.linalg.norm(lbf.model.coef_))


    # Test some values
    test_values = positive_samples[np.random.choice(len(positive_samples), 5, replace=False)]
    test_values2 = negative_samples[np.random.choice(len(negative_samples), 5, replace=False)]
    print("\nPositive Query Results:")
    for x in test_values:
        result = lbf.query(x)
        print(f"Query({x}) = {result}")
    print("\nNegative Query Results:")
    for x in test_values2:
        result = lbf.query(x)
        print(f"Query({x}) = {result}")
