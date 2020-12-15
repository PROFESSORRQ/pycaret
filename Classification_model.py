#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pycaret.utils import version
version()


# In[2]:


from pycaret import *


# In[3]:


import pandas as pd


# In[4]:


newdata='mst2-data.csv'
data_df=pd.read_csv(newdata)
data_df=data_df.dropna()
data_df=data_df.drop_duplicates()


# In[5]:


data_df.shape


# In[12]:


#ans 1
from pycaret.classification import *
setup(data = data_df, target='f4',fold=12,data_split_shuffle=False)
#setup(data = data_df, target = 'cnt',fold=15,normalize=True,normalize_method='zscore',data_split_shuffle=False)
compare_models()


# In[15]:


setup(data = data_df, target = 'f4',fold=12,normalize=True,normalize_method='zscore',remove_outliers=True, outliers_threshold=0.1,data_split_shuffle=False)
compare_models()


# In[ ]:


setup(data = data_df, target = 'cnt',fold=15,normalize=True,normalize_method='zscore',feature_selection = True, feature_selection_threshold = 0.8,data_split_shuffle=False)
compare_models()


# In[11]:


#ans 5
setup(data = data_df, target = 'f4',fold=12,remove_outliers = True, outliers_threshold = 0.15,pca = True, pca_method = 'linear',data_split_shuffle=False)
compare_models()


# In[ ]:




