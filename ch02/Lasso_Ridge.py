
# coding: utf-8

# In[10]:

get_ipython().magic(u'matplotlib inline')
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import mglearn
import numpy as np
import matplotlib.pyplot as plt

X, y = mglearn.datasets.load_extended_boston()
X_train,X_test,y_train,y_test = train_test_split(X, y, random_state = 0)

lasso = Lasso().fit(X_train, y_train)
print ("lasso.intercept_: %s" % lasso.intercept_)
print ("number of features used: %s" % np.sum(lasso.coef_ != 0))

print ("training set score: %s" % lasso.score(X_train,y_train))
print ("test set score: %s" % lasso.score(X_test, y_test))

lasso001 = Lasso(alpha = 0.01).fit(X_train,y_train)
print ("lasso 001 training set score: %s" % lasso001.score(X_train,y_train))  #this model performs well on training set and generalizes better than others
print ("lasso 001 test set score: %s" % lasso001.score(X_test, y_test))

lasso00001 = Lasso(alpha = 0.00001).fit(X_train, y_train)
print ("lasso 00001 training set score: %s" % lasso00001.score(X_train, y_train))  # smaller alpha means more likely to be overfitting
print ("lasso 00001 test set score: %s" % lasso00001.score(X_test,y_test))

ridge01 = Ridge(alpha = 0.1).fit(X_train,y_train)
print ("ridge 01 training set score: %s" % ridge01.score(X_train,y_train))
print ("ridge 01 test set score: %s" % ridge01.score(X_test,y_test))

plt.plot(lasso.coef_,'o',label = "Lasso alpha = 1.0")
plt.plot(lasso001.coef_,'o',label = "Lasso alpha = 0.001")
plt.plot(lasso00001.coef_,'o',label = "Lasso alpha = 0.00001")

plt.plot(ridge01.coef_,'o',label = "Ridge alpha = 0.1")
plt.title("Lasso&Ridge Coefficient")
plt.ylim(-25,25)
plt.legend()

