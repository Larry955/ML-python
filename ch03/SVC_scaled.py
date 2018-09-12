from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state = 1)

svc = SVC(C = 100)
svc.fit(X_train,y_train)
print("Test set accuracy: {:.5f}".format(svc.score(X_test, y_test)))

# preprocessing using 0-1 scaling(MinMaxScaler)
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

svc.fit(X_train_scaled, y_train)
print("MinMAxScaled test set accuracy: {:.5f}".format(svc.score(X_test_scaled, y_test)))

# preprocessing using zero mean and unit variance scaling
std_scaler = StandardScaler()
std_scaler.fit(X_train)
X_train_scaled = std_scaler.transform(X_train)
X_test_scaled = std_scaler.transform(X_test)

svc.fit(X_train_scaled, y_train)
print("StandardScaler test set accuracy: {:.5f}".format(svc.score(X_test_scaled, y_test)))


