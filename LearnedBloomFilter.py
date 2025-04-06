import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. Generate Gaussian data
n_samples = 5000
X = np.random.normal(loc=0.0, scale=1.0, size=(n_samples, 1))

# 2. Define a random interval for the "keys"
a = np.random.uniform(-2, 0)
b = np.random.uniform(0, 2)
if a > b:
    a, b = b, a
print(f"Interval for keys: [{a:.2f}, {b:.2f}]")

# 3. Assign labels based on interval
y = ((X >= a) & (X <= b)).astype(int).flatten()

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 5. Train prefilter (logistic regression)
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. Predict on test set
probs = model.predict_proba(X_test)[:, 1]
tau = 0.5  # Threshold for the prefilter decision
preds = (probs >= tau).astype(int)

# 7. Evaluate
accuracy = accuracy_score(y_test, preds)
print(f"Prefilter accuracy at τ={tau}: {accuracy:.3f}")

# 8. Visualize
plt.figure(figsize=(10, 5))
plt.hist(probs[y_test == 1], bins=50, alpha=0.6, label='Keys (y=1)')
plt.hist(probs[y_test == 0], bins=50, alpha=0.6, label='Non-keys (y=0)')
plt.axvline(tau, color='red', linestyle='--', label=f'Threshold τ={tau}')
plt.title("Prefilter Probability Distribution")
plt.xlabel("Predicted Probability")
plt.ylabel("Count")
plt.legend()
plt.grid(True)
plt.show()