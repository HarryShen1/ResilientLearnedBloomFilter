import numpy as np
from sklearn.linear_model import LogisticRegression
import hashlib

class ConstantPredictor:
    def __init__(self, x):
        self.x = x  

    def predict(self, y):
        return [self.x]

# Simplified, working BloomFilter with deterministic hashing
class BloomFilter:
    def __init__(self, k, m):
        self.k = k
        self.m = m
        self.data = np.zeros((k, m))

    def hash(self, x):
        x_bytes = str(x).encode('utf-8')
        digest = hashlib.sha256(x_bytes).digest()
        return np.array([int.from_bytes(digest[i*4:(i+1)*4], 'big') % self.m for i in range(self.k)])

    def insert(self, x):
        indices = self.hash(x)
        self.data[np.arange(self.k), indices] = 1

    def query(self, x):
        indices = self.hash(x)
        return all(self.data[np.arange(self.k), indices])


class CountingBloomFilter:
    def __init__(self, k, m, data=None):
        self.k = k
        self.m = m
        if data is None:
            self.data = np.zeros((k, m))
        else:
            self.data = data

    def hash(self, x):
        x_bytes = str(x).encode('utf-8')
        digest = hashlib.sha256(x_bytes).digest()
        return np.array([int.from_bytes(digest[i*4:(i+1)*4], 'big') % self.m for i in range(self.k)])

    def insert(self, x):
        indices = self.hash(x)
        self.data[np.arange(self.k), indices] += 1

    def query(self, x):
        indices = self.hash(x)
        return all(self.data[np.arange(self.k), indices])

    def __add__(self, other):
        return CountingBloomFilter(self.k, self.m, self.data + other.data)

    def __neg__(self, other):
        return CountingBloomFilter(self.k, self.m, -self.data)

    def __sub__(self, other):
        return CountingBloomFilter(self.k, self.m, self.data - other.data)



# Learned Bloom Filter
class LearnedBloomFilter:
    def __init__(self, k, m, model):
        self.k = k
        self.m = m
        self.backup_filter = BloomFilter(k, m)
        self.model = model()
        self.trained = False

    # def _get_features(self, x):
    #     indices = self.backup_filter.hash(x)
    #     features = np.zeros(self.m)
    #     features[indices] = 1
    #     return features

    def train(self, positive_samples, negative_samples):
        X_train = []
        y_train = []

        for x in positive_samples:
            X_train.append([x])
            y_train.append(1)

        for x in negative_samples:
            X_train.append([x])
            y_train.append(0)

        if (len(np.unique(y_train)) > 1):
            self.model.fit(X_train, y_train)
        else:
            self.model = ConstantPredictor(y_train[0])
        self.trained = True

        # Selectively add uncertain positives to backup
        inserted = 0
        for x in positive_samples:
            features = x
            prob = self.model.predict([[features]])[0]
            if prob:
                self.backup_filter.insert(x)
                inserted += 1
        print(f"Inserted {inserted} low-confidence positives into backup.")

    def query(self, x):
        if not self.trained:
            raise RuntimeError("Model not trained.")
        features = x
        prob = self.model.predict([[features]])[0]
        if prob:
            return True
        else:
            return self.backup_filter.query(x)


# ----- RUN TEST -----

if __name__ == "__main__":
    # Setup
    k = 100
    m = 100000
    threshold = 0.9
    backup = BloomFilter(k, m)
    lbf = LearnedBloomFilter(k, m, threshold, backup)

    # Toy dataset
    positives = np.array([2.0, 2.1, 2.5, 3.0])
    negatives = np.array([0.0, -0.5, 0.5, 1.0])

    lbf.train(positives, negatives)


    # Test known values
    test_values = [2.0, 0.5, -1.0, 3.0]
    for x in test_values:
        print(f"Query({x}) = {lbf.query(x)}")