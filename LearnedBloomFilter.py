import numpy as np
from sklearn.linear_model import LogisticRegression

# Your BloomFilter class
class BloomFilter:
    def __init__(self, k, p, h, m):
        self.k = k
        self.p = p
        self.h = h
        self.m = m
        self.data = np.zeros((k, m))

    def hash(self, x):
        arr = x ** np.arange(self.k)
        return ((self.h @ arr) % self.p) % self.m

    def insert(self, x):
        indices = self.hash(x)
        self.data[np.arange(self.k), indices] = 1

    def query(self, x):
        indices = self.hash(x)
        indices = indices.astype(int)  # Ensure indices are integers
        return all(self.data[np.arange(self.k), indices])


# Your Learned Bloom Filter class
class LearnedBloomFilter:
    def __init__(self, k, p, h, m, backup_filter=None):
        self.k = k
        self.p = p
        self.h = h
        self.m = m
        self.backup_filter = backup_filter
        self.model = LogisticRegression()  # Using logistic regression as a simple model
        self.trained = False

    def train(self, positive_samples, negative_samples):
        # Prepare training data for the model
        X_train = []
        y_train = []

        # Generate features for each sample (hash positions)
        for sample in positive_samples:
            indices = self.backup_filter.hash(sample)
            indices = indices.astype(int)  # Ensure indices are integers
            features = np.zeros(self.m)
            features[indices] = 1
            X_train.append(features)
            y_train.append(1)  # Positive class

        for sample in negative_samples:
            indices = self.backup_filter.hash(sample)
            indices = indices.astype(int)  # Ensure indices are integers
            features = np.zeros(self.m)
            features[indices] = 1
            X_train.append(features)
            y_train.append(0)  # Negative class

        # Train the logistic regression model
        self.model.fit(X_train, y_train)
        self.trained = True

    def query(self, x):
        if not self.trained:
            raise ValueError("Model is not trained yet.")

        # Get hash positions and create feature vector
        indices = self.backup_filter.hash(x)
        indices = indices.astype(int)  # Ensure indices are integers
        features = np.zeros(self.m)
        features[indices] = 1

        # Use the trained model to make a prediction
        prediction = self.model.predict([features])[0]

        # If model is not confident (e.g., 0.5 threshold), fall back to backup filter
        if prediction == 1:
            return True
        else:
            return self.backup_filter.query(x)


# --- Test with Gaussian Distribution ---

# Initialize your BloomFilter (Backup filter)
k = 100
p = 2 ** 61 - 1
h = np.vectorize(int)(np.random.rand(k, k) * (p + 1))  # Random hash functions
m = 100000

backup_filter = BloomFilter(k, p, h, m)

# Initialize Learned Bloom Filter with the backup filter
learned_filter = LearnedBloomFilter(k, p, h, m, backup_filter)

# Generate Positive and Negative Samples from a Gaussian distribution
mu = 0  # Mean of the Gaussian
sigma = 1  # Standard deviation
n = 10000  # Number of samples

# Generate samples
samples = np.random.normal(loc=mu, scale=sigma, size=n)

# Define positive samples as values greater than 1 (for example)
positive_samples = samples[samples > 1]

# Define negative samples as values less than or equal to 1
negative_samples = samples[samples <= 1]

# Train the Learned Bloom Filter with positive and negative samples
learned_filter.train(positive_samples, negative_samples)

# Query the Learned Bloom Filter with new values
test_values = [2, 0.5, -0.5, 3, -2]

# Test the Learned Bloom Filter on a few samples
results = {x: learned_filter.query(x) for x in test_values}

print("Query Results:")
for value, result in results.items():
    print(f"Value: {value}, Learned Bloom Filter result: {result}")
