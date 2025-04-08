import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


# --- BloomFilter Class ---
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
        indices = indices.astype(int)  # Ensure indices are integers
        self.data[np.arange(self.k), indices] = 1

    def query(self, x):
        indices = self.hash(x)
        indices = indices.astype(int)  # Ensure indices are integers
        return all(self.data[np.arange(self.k), indices])


# --- Sandwiched Learned Bloom Filter Class ---
class SandwichedLearnedBloomFilter:
    def __init__(self, k, p, h, m, backup_filter_1=None, backup_filter_2=None):
        self.k = k
        self.p = p
        self.h = h
        self.m = m
        self.backup_filter_1 = backup_filter_1  # Fast Bloom Filter (1st)
        self.backup_filter_2 = backup_filter_2  # Accurate Bloom Filter (2nd)
        self.model = LogisticRegression()  # Machine learning model for optimization
        self.trained = False

    def train(self, positive_samples, negative_samples):
        # Prepare training data for the model
        X_train = []
        y_train = []

        # Generate features for each sample (hash positions from both bloom filters)
        for sample in positive_samples:
            indices_1 = self.backup_filter_1.hash(sample)
            indices_2 = self.backup_filter_2.hash(sample)
            indices_1 = indices_1.astype(int)
            indices_2 = indices_2.astype(int)

            # Create a feature vector for each Bloom filter separately
            features_1 = np.zeros(self.m)
            features_1[indices_1] = 1

            features_2 = np.zeros(self.m)
            features_2[indices_2] = 1

            # Combine the features from both Bloom filters
            X_train.append(np.concatenate([features_1, features_2]))
            y_train.append(1)  # Positive class

        for sample in negative_samples:
            indices_1 = self.backup_filter_1.hash(sample)
            indices_2 = self.backup_filter_2.hash(sample)
            indices_1 = indices_1.astype(int)
            indices_2 = indices_2.astype(int)

            # Create a feature vector for each Bloom filter separately
            features_1 = np.zeros(self.m)
            features_1[indices_1] = 1

            features_2 = np.zeros(self.m)
            features_2[indices_2] = 1

            # Combine the features from both Bloom filters
            X_train.append(np.concatenate([features_1, features_2]))
            y_train.append(0)  # Negative class

        # Train the logistic regression model
        self.model.fit(X_train, y_train)
        self.trained = True

    def query(self, x):
        if not self.trained:
            raise ValueError("Model is not trained yet.")

        # Get hash positions and create feature vector from both Bloom Filters
        indices_1 = self.backup_filter_1.hash(x)
        indices_2 = self.backup_filter_2.hash(x)
        indices_1 = indices_1.astype(int)
        indices_2 = indices_2.astype(int)

        # Create feature vector for query
        features_1 = np.zeros(self.m)
        features_1[indices_1] = 1

        features_2 = np.zeros(self.m)
        features_2[indices_2] = 1

        features = np.concatenate([features_1, features_2])

        # Use the trained model to make a prediction
        prediction = self.model.predict([features])[0]

        # If model is confident (prediction = 1), return True, otherwise fall back to Bloom Filter 2
        if prediction == 1:
            return True
        else:
            return self.backup_filter_2.query(x)


# --- Example Usage ---
# Initialize your BloomFilter (Backup filter 1 and Backup filter 2)
k = 100
p = 2 ** 61 - 1
h = np.vectorize(int)(np.random.rand(k, k) * (p + 1))  # Random hash functions
m = 100000

# Create two Bloom Filters: One for fast queries and the other for more accurate membership
backup_filter_1 = BloomFilter(k, p, h, m)
backup_filter_2 = BloomFilter(k, p, h, m)

# Initialize Sandwiched Learned Bloom Filter with both backup filters
sandwiched_filter = SandwichedLearnedBloomFilter(k, p, h, m, backup_filter_1, backup_filter_2)

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

# Train the Sandwiched Learned Bloom Filter with positive and negative samples
sandwiched_filter.train(positive_samples, negative_samples)

# Query the Sandwiched Learned Bloom Filter with new values
test_values = [2, 0.5, -0.5, 3, -2]

# Test the Sandwiched Learned Bloom Filter on a few samples
results = {x: sandwiched_filter.query(x) for x in test_values}

print("Query Results:")
for value, result in results.items():
    print(f"Value: {value}, Sandwiched Bloom Filter result: {result}")

# Visualization: Plot histogram of the positive samples
plt.hist(positive_samples, bins=50, density=True, alpha=0.7, color='skyblue', label='Positive Samples')
plt.hist(negative_samples, bins=50, density=True, alpha=0.7, color='salmon', label='Negative Samples')
plt.title("Positive and Negative Samples from Gaussian Distribution")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()
