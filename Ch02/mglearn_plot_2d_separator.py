
# coding: utf-8

# In[26]:

get_ipython().magic(u'matplotlib inline')
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import mglearn
cancer = load_breast_cancer()
#X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target)
print X_train.shape,X_test.shape,cancer.data.shape

X,y = mglearn.datasets.make_forge()
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0)
clf = KNeighborsClassifier(n_neighbors = 3)
clf.fit(X_train,y_train)
clf.predict(X_test)
clf.score(X_test,y_test)

fig,axes = plt.subplots(1,3,figsize = (15,5))
for n_neighbors,ax in zip([1,3,9],axes):
    clf = KNeighborsClassifier(n_neighbors = n_neighbors).fit(X,y)
    mglearn.plots.plot_2d_separator(clf, X, fill = True, eps = 0.5, ax = ax, alpha = .4)
    ax.scatter(X[:,0],X[:,1],c = y ,s = 60, cmap = mglearn.cm2)
    ax.set_title("%d neighbor(s)" % n_neighbors)

