#!/usr/bin/env python
# coding: utf-8

# 
# 

# In[2]:


from pycaret.utils import version
version()


# In[3]:


from pycaret.datasets import get_data


# In[4]:


# Internet connection is required
dataSets = get_data('index')
dataSets


# In[5]:


# Internet connection is required
bike_df = get_data("bike")
# This is regression dataset. The values in medv are continuous values


# In[6]:


print(bike_df.shape)


# In[7]:


print(bike_df.shape)
bike_df.drop_duplicates()
print(bike_df.shape)


# ### <span style='color:DarkBlue'>1.2 Parameter setting for all regression models</span>
# - Train/Test division
# - Sampling
# - Normalization
# - Transformation
# - PCA (Dimention Reduction)
# - Handaling of Outliers
# - Feature Selection

# In[8]:


from pycaret.regression import *
reg = setup(data = bike_df, target='cnt',fold=15,normalize=True,normalize_method='zscore',data_split_shuffle=False)
compare_models()


# In[11]:


compare_models()
# Explore more parameters


# In[ ]:


ExtraTreesModel = create_model('et')
plot_model(ExtraTreesModel, plot='residuals')


# In[ ]:


plot_model(ExtraTreesModel, plot='error')


# In[ ]:


plot_model(ExtraTreesModel, plot='learning')


# In[ ]:


# Take long time and may show error
#plot_model(catboostModel, plot='vc')


# In[16]:


setup(data = bike_df, target = 'cnt',fold=15,normalize=True,normalize_method='zscore',data_split_shuffle=False)
compare_models()

#normalize_method = {zscore, minmax, maxabs, robust}


# In[19]:


setup(data = boston_df, target = 'cnt', feature_selection = True, feature_selection_threshold = 0.8)
compare_models()


# In[ ]:


setup(data = boston_df, target = 'medv', remove_outliers = True, outliers_threshold = 0.05)
compare_models()


# In[23]:


setup(data = bike_df, target = 'cnt', remove_outliers = True, outliers_threshold = 0.1,fold=15,normalize=True,normalize_method='zscore',data_split_shuffle=False)
compare_models()


# In[22]:


setup(data = bike_df, target = 'cnt',fold=15, pca = True, pca_method = 'linear', remove_outliers = True, outliers_threshold = 0.15,data_split_shuffle=False)
compare_models()


# In[ ]:


reg_model_et = create_model('et', fold=15)
compare_models()
# Explore more parameters


# In[ ]:


# Create Other Models
Linear Regression             'lr'                   linear_model.LinearRegression
Lasso Regression              'lasso'                linear_model.Lasso
Ridge Regression              'ridge'                linear_model.Ridge
Elastic Net                   'en'                   linear_model.ElasticNet
Least Angle Regression        'lar'                  linear_model.Lars
Lasso Least Angle Regression  'llar'                 linear_model.LassoLars
Orthogonal Matching Pursuit   'omp'                  linear_model.OMP
Bayesian Ridge                'br'                   linear_model.BayesianRidge
Automatic Relevance Determ.   'ard'                  linear_model.ARDRegression
Passive Aggressive Regressor  'par'                  linear_model.PAR
Random Sample Consensus       'ransac'               linear_model.RANSACRegressor
TheilSen Regressor            'tr'                   linear_model.TheilSenRegressor
Huber Regressor               'huber'                linear_model.HuberRegressor 
Kernel Ridge                  'kr'                   kernel_ridge.KernelRidge
Support Vector Machine        'svm'                  svm.SVR
K Neighbors Regressor         'knn'                  neighbors.KNeighborsRegressor 
Decision Tree                 'dt'                   tree.DecisionTreeRegressor
Random Forest                 'rf'                   ensemble.RandomForestRegressor
Extra Trees Regressor         'et'                   ensemble.ExtraTreesRegressor
AdaBoost Regressor            'ada'                  ensemble.AdaBoostRegressor
Gradient Boosting             'gbr'                  ensemble.GradientBoostingRegressor 
Multi Level Perceptron        'mlp'                  neural_network.MLPRegressor
Extreme Gradient Boosting     'xgboost'              xgboost.readthedocs.io
Light Gradient Boosting       'lightgbm'             github.com/microsoft/LightGBM
CatBoost Regressor            'catboost'             https://catboost.ai


# In[ ]:


reg_model_catboost_tuned = tune_model(reg_model_catboost, n_iter=10, optimize = 'mae')
# Explore more parameters


# In[ ]:


save_model(reg_model_catboost_tuned, 'CatBoostModel')


# In[ ]:


data = get_data("boston")


# In[ ]:


# Select top 10 rows
new_data = data.iloc[:10]
new_data


# In[ ]:


newPredictions = predict_model(CatBoostModel, data = new_data)
newPredictions


# In[ ]:


import matplotlib.pyplot as plt
actual = newPredictions.iloc[:,-2]
predicted = newPredictions.iloc[:,-1]
plt.scatter(actual, predicted)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Actul Vs Predicted')
plt.savefig("result-scatter-plot-lss.jpg", dpi=300)
plt.show()


# In[ ]:


newPredictions.to_csv("NewPredictions.csv")
# Email the result to the user


# In[ ]:


Residuals Plot               'residuals'
Prediction Error Plot        'error'
Cooks Distance Plot          'cooks'
Recursive Feat. Selection    'rfe'
Learning Curve               'learning'
Validation Curve             'vc'
Manifold Learning            'manifold'
Feature Importance           'feature'
Model Hyperparameter         'parameter'


# In[ ]:


rf = create_model('rf')


# In[ ]:


plot_model(rf, plot='residuals')


# In[ ]:


plot_model(rf, plot='error')


# In[ ]:


plot_model(rf, plot='cooks')


# In[ ]:


# Take 3-4 minutes
# plot_model(rf, plot='rfe')


# In[ ]:


plot_model(rf, plot='learning')


# In[ ]:


plot_model(rf, plot='vc')


# In[ ]:


plot_model(rf, plot='manifold')


# In[ ]:


plot_model(rf, plot='parameter')


# In[ ]:


model = create_model('catboost')
plot_model(model, plot='feature')


# In[ ]:


model = create_model('et')
plot_model(model, plot='feature')


# In[ ]:


model = create_model('lightgbm')
plot_model(model, plot='feature')


# In[ ]:


model = create_model('gbr')
plot_model(model, plot='feature')


# In[ ]:


model = create_model('xgboost')
plot_model(model, plot='feature')


# In[ ]:


model = create_model('rf')
plot_model(model, plot='feature')

