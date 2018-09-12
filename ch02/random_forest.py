
# coding: utf-8

# In[27]:

get_ipython().magic(u'matplotlib inline')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import mglearn
import numpy as np

X,y = make_moons(n_samples = 100, noise = 0.25, random_state = 3)
X_train,X_test, y_train,y_test = train_test_split(X,y,stratify = y, random_state = 42)

forest = RandomForestClassifier(n_estimators = 5, random_state = 42)  # five trees in this forest
forest.fit(X_train,y_train)

RandomForestClassifier(bootstrap = True, class_weight = None, criterion = 'gini',  # gini gain
                      max_depth = None, max_features = 'auto',  # when set to auto, max_features = sqrt(forest.n_features_)
                      min_samples_leaf = 1, min_samples_split = 2,
                      min_weight_fraction_leaf = 0.0, n_estimators = 5, n_jobs = 1,
                      oob_score = False, random_state = 2, verbose = 0, warm_start = False)

fig, axes = plt.subplots(2,3,figsize = (20,10))
for i, (ax,tree) in enumerate(zip(axes.ravel(),forest.estimators_)):
    ax.set_title("tree %d" % i)
    mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax = ax)
mglearn.plots.plot_2d_separator(forest, X_train, fill = True, ax = axes[-1,-1], alpha=.4)
axes[-1,-1].set_title("random forest")
plt.scatter(X_train[:,0],X_train[:,1], c = np.array(['r','b'])[y_train], s = 80)




# In[62]:

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,random_state = 0)
forest = RandomForestClassifier(n_estimators=100,random_state=0).fit(X_train,y_train)

print ("training set score: %s" % forest.score(X_train,y_train))
print ("test set score: %s" % forest.score(X_test,y_test))


plt.figure(figsize=(15,8))
plt.plot(forest.feature_importances_, 'o')
plt.xticks(range(cancer.data.shape[1]),cancer.feature_names, rotation = 90,fontsize = 20)
plt.ylim(0,.2)
plt.suptitle("feature importances", fontsize = 20)

