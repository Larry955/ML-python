
# coding: utf-8

# In[11]:

get_ipython().magic(u'matplotlib inline')
import mglearn
import matplotlib.pyplot as plt

plt.figure(figsize=(15,10))
mglearn.plots.plot_animal_tree()

plt.suptitle("animal_tree")

mglearn.plots.plot_tree_progressive()


# In[46]:

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

cancer = load_breast_cancer()
X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,stratify = cancer.target,random_state = 42)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train,y_train)
print ("accuracy on training set: %s" % tree.score(X_train,y_train))
print ("accuracy on test set: %s" % tree.score(X_test,y_test))

tree_depth4 = DecisionTreeClassifier(max_depth=4,random_state=0).fit(X_train,y_train)
print ("accuracy on training set: %s" % tree_depth4.score(X_train,y_train))
print ("accuracy on test set: %s" % tree_depth4.score(X_test,y_test))

tree_depth5 = DecisionTreeClassifier(max_depth=5,random_state=0).fit(X_train,y_train)
print ("accuracy on training set: %s" % tree_depth5.score(X_train,y_train))
print ("accuracy on test set: %s" % tree_depth5.score(X_test,y_test))

tree = mglearn.plots.plot_tree_not_monotone()
plt.suptitle("tree_not_monotone")


# In[21]:

from sklearn.tree import export_graphviz
import graphviz
export_graphviz(tree,out_file="mytree.dot",class_names = ["malignant", "benign"], 
                feature_names = cancer.feature_names,impurity = False, filled = True)
with open("mytree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)


# In[42]:

plt.figure(figsize=(10,10))
plt.plot(tree.feature_importances_,'*')
plt.xticks(range(cancer.data.shape[1]),cancer.feature_names,rotation = 90)
plt.ylim(-.01,1)

