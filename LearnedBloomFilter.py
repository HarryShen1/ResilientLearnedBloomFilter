import numpy as np
from sklearn.linear_model import LogisticRegression
import hashlib

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
    def __init__(self, k, m, confidence_threshold, backup_filter):
        self.k = k
        self.m = m
        self.confidence_threshold = confidence_threshold
        self.backup_filter = backup_filter
        self.model = LogisticRegression()
        self.trained = False

    def _get_features(self, x):
        indices = self.backup_filter.hash(x)
        features = np.zeros(self.m)
        features[indices] = 1
        return features

    def train(self, positive_samples, negative_samples):
        X_train = []
        y_train = []

        for x in positive_samples:
            X_train.append(self._get_features(x))
            y_train.append(1)

        for x in negative_samples:
            X_train.append(self._get_features(x))
            y_train.append(0)

        self.model.fit(X_train, y_train)
        self.trained = True

        # Selectively add uncertain positives to backup
        inserted = 0
        for x in positive_samples:
            features = self._get_features(x)
            prob = self.model.predict_proba([features])[0][1]
            if prob < self.confidence_threshold:
                self.backup_filter.insert(x)
                inserted += 1
        print(f"Inserted {inserted} low-confidence positives into backup.")

    def query(self, x):
        if not self.trained:
            raise RuntimeError("Model not trained.")
        features = self._get_features(x)
        prob = self.model.predict_proba([features])[0][1]
        if prob >= self.confidence_threshold:
            return True
        else:
            return self.backup_filter.query(x)


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
