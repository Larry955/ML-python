
# coding: utf-8

# In[3]:

import numpy as np
np_array = [[1,2,3],[3,4,5]]
print np_array


# In[8]:

eye = np.eye(3)
print eye

from scipy import sparse
sparse_eye = sparse.csr_matrix(eye)
print sparse_eye


# In[21]:

from sklearn.datasets import load_iris
iris = load_iris()
iris.keys()
print (iris['feature_names'])
print (iris['target_names'])
print (iris['target'])
print (iris['data'])


# In[24]:

type(iris['feature_names'])


# In[28]:

iris['data'].shape


# In[31]:

iris['target_names'].shape


# In[32]:

iris['data'][:5]


# In[49]:

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(iris['data'],iris['target'])
print (X_train)
print (X_test.shape)
print (y_train.shape)
print (iris['data'].shape[0])


# In[52]:

print iris['feature_names']


# In[3]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()
X_train,X_test,y_train,y_test = train_test_split(iris['data'], iris['target'])
fig,ax = plt.subplots(3,3,figsize = (15,15))
plt.suptitle('iris_pairplot')
for i in range(3):
    for j in range(3):
        ax[i,j].scatter(X_train[:,j],X_train[:,i + 1],c = y_train, s = 60)
        ax[i,j].set_xticks(())
        ax[i,j].set_yticks(())
        if i == 2:
            ax[i,j].set_xlabel(iris['feature_names'][j])
        if j == 0:
            ax[i,j].set_ylabel(iris['feature_names'][i + 1])
        if j > i:
            ax[i,j].set_visible(False)


# In[ ]:



