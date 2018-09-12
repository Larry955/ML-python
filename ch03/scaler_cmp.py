from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import mglearn

# make synthetic data
X, _ = make_blobs(n_samples = 50, centers = 5, random_state = 4, cluster_std = 2)
# split it into training and test sets
X_train, X_test = train_test_split(X, random_state = 5, test_size = .1)
print("X_train shape: {}".format(X_train.shape))

# plot the training and test sets
fig, axes = plt.subplots(1, 3, figsize = (13, 4))

axes[0].scatter(X_train[:, 0], X_train[:, 1], c = mglearn.cm2(0), label = "Training set", s = 60)
axes[0].scatter(X_test[:, 0], X_test[:, 1], c = mglearn.cm2(1), label = "Test set", s = 60)
axes[0].legend(loc = "best")
axes[0].set_title("Original Data")

# scale the data using MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# visualize the properly scaled data
axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c = mglearn.cm2(0), label = "Training set", s = 60)
axes[1].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c = mglearn.cm2(1), label = "Test set", s = 60)
axes[1].set_title("Scaled Data")

# Bad behavior: rescale the test set separately
# DO NOT DO THIS! for illustration purposes only
test_scaler = MinMaxScaler()
test_scaler.fit(X_test)
X_test_scaled_badly = test_scaler.transform(X_test)

axes[2].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c = mglearn.cm2(0), label = "Training set", s = 60)
axes[2].scatter(X_test_scaled_badly[:, 0], X_test_scaled_badly[:, 1], c = mglearn.cm2(1), label = "Test set", s = 60)
axes[2].set_title("Improperly Scaled Data")

for ax in axes:
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")

plt.show()
