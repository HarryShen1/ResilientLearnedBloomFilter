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


# ----- RUN TEST -----

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