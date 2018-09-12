
# coding: utf-8

# In[6]:

import mglearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_wave()
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0)
lr = LinearRegression().fit(X_train,y_train)

print ("lr.coef_: %s" % lr.coef_)
print ("lr.intercept_: %s" % lr.intercept_)

print ("training set score: %s" % lr.score(X_train, y_train))
print ("test set score: %s" % lr.score(X_test,y_test))


# In[17]:

X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 0)
lr = LinearRegression().fit(X_train,y_train)

print ("lr.coef_.shape: %s" % lr.coef_.shape)
print ("lr.intercept_: %s" % lr.intercept_)
print ("training set score: %s" % lr.score(X_train,y_train))
print ("test set score: %s" % lr.score(X_test,y_test))      #overfitting


# In[29]:

get_ipython().magic(u'matplotlib inline')
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

ridge = Ridge().fit(X_train,y_train)   # default: reguralization factor alpha = 1
print ("training set score: %s" % ridge.score(X_train,y_train))
print ("test set score: %s" % ridge.score(X_test,y_test))   #works well

ridge10 = Ridge(alpha = 10).fit(X_train,y_train)
print ("\ntraining set score: %s" % ridge10.score(X_train, y_train))
print ("test set score: %s" % ridge10.score(X_test, y_test))

ridge01 = Ridge(alpha = 0.1).fit(X_train,y_train)     # smaller alpha means less restriction, and is more likely to be overfitting, it becomes Linear Regression when alpha = 0
print ("\ntraining set score: %s" % ridge01.score(X_train,y_train))
print ("test set score: %s" % ridge01.score(X_test,y_test))

plt.title("ridge_coefficients")
plt.plot(ridge.coef_, 'o',label = "Ridge alpha = 1.0")
plt.plot(ridge10.coef_, 'o',label = "Ridge alpha = 10")
plt.plot(ridge01.coef_, 'o',label = "Ridge alpha = 0.1") 

plt.plot(lr.coef_,'o',label = "Linear Regression")
plt.ylim(-25,25)
plt.legend(loc = "upper right")


# In[ ]:



