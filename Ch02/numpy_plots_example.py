
# coding: utf-8

# In[1]:

print 1


# In[2]:

import numpy as np
x = np.array([[1,2,3],[2,3,4]])
print x


# In[3]:

from scipy import sparse
# create a 2d array with a diagonal of ones, and zeros everywhere else
eye = np.eye(4)
print ("Numpy array:\n%s" % eye)


# In[6]:

#convert the numpy array into a scipy sparse matrix in CSR format
#only non-zero entries are stored
sparse_matrix = sparse.csr_matrix(eye)
print("\nthe sparse CSR matrix: \n%s " % sparse_matrix)


# In[25]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt

#generate a sequences of integers
x = np.arange(20)
#create a second array using sin function
y = np.sin(x)
#the plot function makes a line chart of one array against another
plt.plot(x,y,marker = "o",color = "red")


# In[34]:

import pandas as pd
#create a simple dataset of people
data = {
        'Name': ["John","Tony","Mary"],
       'Location':["New York","Paris","London"],
       'Age':[1,4,54]
        }
pandas_data = pd.DataFrame(data)
pandas_data
print ("\npandas version:%s" % pd.__version__)

