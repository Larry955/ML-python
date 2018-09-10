
# coding: utf-8

# In[13]:

get_ipython().magic(u'matplotlib inline')
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mglearn

cancer = load_breast_cancer()
X_train,X_test,y_train,y_test = train_test_split(cancer.data, cancer.target,random_state = 0)

gbrt = GradientBoostingClassifier(random_state = 0).fit(X_train,y_train)   # default: max_depth = 100, learning_rate = 0.1
print ("training set score :%s" % gbrt.score(X_train,y_train))
print ("test set score: %s" % gbrt.score(X_test, y_test))

gbrt_max_depth_1 = GradientBoostingClassifier(random_state=0, max_depth=1).fit(X_train,y_train)
print ("training set score: %s" % gbrt_max_depth_1.score(X_train,y_train))
print ("test set score: %s" % gbrt_max_depth_1.score(X_test,y_test))


gbrt_learning_rate_001 = GradientBoostingClassifier(random_state=0, learning_rate=0.01).fit(X_train,y_train)
print ("training set score: %s" % gbrt_learning_rate_001.score(X_train,y_train))
print ("test set score: %s" %gbrt_learning_rate_001.score(X_test,y_test))

plt.figure(figsize = (10,8))
plt.plot(gbrt_max_depth_1.feature_importances_,'o')
plt.xticks(range(cancer.data.shape[1]),cancer.feature_names,rotation = 90)
plt.suptitle("feature importances")

