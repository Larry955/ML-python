
# coding: utf-8

# In[24]:

get_ipython().magic(u'matplotlib inline')
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import mglearn
import matplotlib.pyplot as plt

X,y = mglearn.datasets.make_forge()
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0)
clf = KNeighborsClassifier(n_neighbors = 3)
clf.fit(X_train,y_train)
clf.predict(X_test)
clf.score(X_test,y_test)

fig, axes = plt.subplots(1,3,figsize = (10,3))
for n_neighbors,ax in zip([1,3,9],axes):
    clf = KNeighborsClassifier(n_neighbors = n_neighbors).fit(X,y)
    mglearn.plots.plot_2d_separator(clf, X, fill = True, eps = 0.5, ax = ax, alpha = .3)
    ax.scatter(X[:,0],X[:,1],c = y, s = 60,cmap = mglearn.cm2)
    ax.set_title("%d neighbor(s)" % n_neighbors)
mglearn.plots.plot_knn_regression(n_neighbors = 3)


# In[63]:

get_ipython().magic(u'matplotlib inline')
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
X,y = mglearn.datasets.make_wave(n_samples = 40)

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0)

reg = KNeighborsRegressor(algorithm = 'auto', leaf_size = 30, metric = 'minkowski',
                   metric_params = None, n_jobs = 1,n_neighbors = 3, p = 2,
                   weights = 'uniform')
reg.fit(X_train,y_train)

reg.predict(X_test)
reg.score(X_test,y_test)

fig, axes = plt.subplots(1,3,figsize = (15,4))
#create 1000 data points, evenly spaced between -3 and 3
line = np.linspace(-3, 3 , 1000).reshape(-1,1)
plt.suptitle("nearest_neighbors_regression")
for n_neighbors, ax in zip([1,3,9], axes):
    # make predictions using 1, 3 or 9 neighbors
    reg = KNeighborsRegressor(n_neighbors = n_neighbors).fit(X,y)
    ax.plot(X, y, 'o', color = 'red')
    ax.plot(X, -3 * np.ones(len(X)), 'o', color = 'green')
    ax.plot(line, reg.predict(line), color = 'blue')
    ax.set_title("%d neighbor(s)" % n_neighbors)
    

