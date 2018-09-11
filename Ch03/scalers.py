import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

cancer = load_breast_cancer()
X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,random_state = 1)

print(X_train.shape)
print(X_test.shape)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
# print dataset properties before and after scaling
print("transformed shaped: {}".format(X_train_scaled.shape))
print("per-feature minimum before scaling: {}".format(X_train.min(axis = 0)))
print("per-feature maximum before scaling: {}".format(X_train.max(axis = 0)))
print("per-feature minimum after scaling: {}".format(X_train_scaled.min(axis = 0)))
print("per-feature maximum after scaling: {}".format(X_train_scaled.max(axis = 0)))
mglearn.plots.plot_scaling()

plt.show()
