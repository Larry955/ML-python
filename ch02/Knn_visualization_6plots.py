
# coding: utf-8

# In[14]:

get_ipython().magic(u'matplotlib inline')
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()
# print iris['feature_names']
X_train,X_test,y_train,y_test = train_test_split(iris['data'],iris['target'])
fig, ax = plt.subplots(3,3,figsize = (15,15))
plt.suptitle('iris_pairplot')

# for i in range(3):
#     print (X_train[:,i])
for i in range(3):
    for j in range(3):
        ax[i][j].scatter(X_train[:,j],X_train[:,i + 1], c = y_train, s = 60)
        if i == 2:
            ax[i][j].set_xlabel(iris['feature_names'][j])
        if j == 0:
            ax[i][j].set_ylabel(iris['feature_names'][i + 1])

knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train,y_train)
knn.score(X_test,y_test)



# In[81]:

import mglearn

X,y = mglearn.datasets.make_forge()
#plt.scatter(X[:,0],X[:,1], c = y)
mglearn.plots.plot_knn_classification(n_neighbors = 3)
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0)
clf = KNeighborsClassifier(n_neighbors =3)
clf.fit(X_train,y_train)
clf.predict(X_test)
clf.score(X_test,y_test)


# In[54]:

from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston
cancer = load_breast_cancer()
boston = load_boston()
print boston.feature_names,boston.DESCR

