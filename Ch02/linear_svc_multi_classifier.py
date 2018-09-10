
# coding: utf-8

# In[10]:

get_ipython().magic(u'matplotlib inline')
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import mglearn
import matplotlib.pyplot as plt

X,y = mglearn.datasets.make_forge()

fig, axes = plt.subplots(1,2,figsize = (10,3))
plt.suptitle("linear_classifiers")

for model, ax in zip([LinearSVC(),LogisticRegression()],axes):
    clf = model.fit(X,y)
    mglearn.plots.plot_2d_separator(clf,X,fill = False, eps = 0.5, ax = ax, alpha = 0.7)
    ax.scatter(X[:,0],X[:,1],c = y, s = 60, cmap = mglearn.cm2)
    ax.set_title("%s" % clf.__class__.__name__)
    
mglearn.plots.plot_linear_svc_regularization()   # larger C means overfitting, which is contrast with alpha in linear regression 


# In[49]:

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,stratify = cancer.target, random_state = 42)
logistic_regression = LogisticRegression().fit(X_train,y_train)
print ("training set score: %s" % logistic_regression.score(X_train,y_train))
print ("test set score: %s" % logistic_regression.score(X_test,y_test))

logistic_regression_100 = LogisticRegression(C = 100).fit(X_train,y_train)
print ("training set score: %s" % logistic_regression_100.score(X_train,y_train))
print ("test set score: %s" % logistic_regression_100.score(X_test,y_test))

logistic_regression_001 = LogisticRegression(C = 0.01).fit(X_train,y_train)
print ("training set score: %f" % logistic_regression_001.score(X_train,y_train))
print ("test set score: %f\n" % logistic_regression_001.score(X_test,y_test))

# plt.plot(logistic_regression.coef_.T,'o',label = "C = 1")
# plt.plot(logistic_regression_100.coef_.T,'o',label = "C = 100")
# plt.plot(logistic_regression_001.coef_.T,'o',label = "C = 0.01")
# plt.xticks(range(cancer.data.shape[1]),cancer.feature_names,rotation = 90)
# plt.ylim(-5,5)
# plt.legend()

for C in [0.001,1.0,100]:
    lr_l1 = LogisticRegression(C = C, penalty="l1").fit(X_train,y_train)
    print ("training set score with C = %f: %f" % (C, lr_l1.score(X_train,y_train)))
    print ("test set score with C = %f: %f" % (C, lr_l1.score(X_test,y_test)))
    plt.plot(lr_l1.coef_.T, 'o',label = "C = %f" % C)
    plt.xticks(range(cancer.data.shape[1]),cancer.feature_names, rotation = 90)
plt.ylim(-5,5)
plt.legend(loc = "upper right")


# In[94]:

from sklearn.datasets import make_blobs
import numpy as np

X, y = make_blobs(random_state = 42)
plt.scatter(X[:, 0], X[:, 1], c = y, s = 60, cmap = mglearn.cm3)
linear_svc = LinearSVC().fit(X,y)

# line = np.linspace(-15,15)
# for coef, intercept in zip(linear_svc.coef_,linear_svc.intercept_):
#     plt.plot(line, -(line * coef[0] + intercept) / coef[1])
# plt.ylim(-10,15)
# plt.xlim(-10,8)


mglearn.plots.plot_2d_classification(linear_svc,X,fill=True,alpha=.6)
plt.scatter(X[:, 0], X[:, 1], c = y, s = 60,cmap = mglearn.cm3)
line = np.linspace(-15,15)
for coef, intercept in zip(linear_svc.coef_,linear_svc.intercept_):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1])
    

