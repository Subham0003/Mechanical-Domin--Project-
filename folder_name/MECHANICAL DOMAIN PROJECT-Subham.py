#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('nims.csv')


# In[3]:


df.head()


# In[9]:


df.drop(["Sl. No."],axis=1,inplace=True)


# In[10]:


df.head()


# In[5]:





# In[6]:


df


# In[7]:


df.info()


# In[8]:


df.describe().T


# In[45]:


import pandas as pd
from pandas_profiling import ProfileReport

# Load your DataFrame
df = pd.read_csv('nims.csv')

# Create a Profile Report
profile = ProfileReport(df, title='Pandas Profiling Report', explorative=True)

# Display the Report in the Notebook
profile.to_widgets()

# Optionally, save the report to an HTML file
profile.to_file("subham_pandas_profiling_report.html")


# In[9]:


def boxplots(col):
    sns.boxplot(df[col])
    plt.show()
    
for i in list(df.select_dtypes(exclude=['object']).columns)[0:]:
              boxplots(i)


# In[10]:


plt.figure(figsize=(20,15))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
plt.show()


# In[13]:


df = df.drop(['CT', 'Tt','THt'], axis=1)


# In[14]:


df = df.drop(['THT'], axis=1)


# In[15]:


df = df.drop(['TCr'], axis=1)


# In[16]:


df = df.drop(['NT'], axis=1)


# In[17]:


df = df.drop(['DT'], axis=1)


# In[18]:


df = df.drop(['THQCr'], axis=1)


# In[19]:


"""
import pandas_profiling as pp
report = pp.ProfileReport(df)
"""


# In[20]:


x = df.iloc[:,:-1]
y = df.iloc[:,-1]


# In[21]:


x.head()


# In[53]:


y.head()


# In[22]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_sc = sc.fit_transform(x)


# In[26]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(x_train, y_train)


# In[27]:


feature_importances = rfr.feature_importances_


# In[28]:


feature_importances 


# In[34]:


feature_names = list(x.columns)


# In[35]:


feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})


# In[36]:


feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)


# In[37]:


print(feature_importance_df)


# In[38]:


plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance for Random Forest')
plt.show()


# In[ ]:





# # Variance Inflaction Factor

# In[24]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
variable = x_sc
vif = pd.DataFrame()

vif['vif'] = [variance_inflation_factor(variable, i) for i in range(variable.shape[1])]
vif['Features'] = x.columns
vif


# In[ ]:


# PCA 
# Ensemble method


# In[25]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_sc, y, test_size=0.25, random_state=101)


# # Linear Regression

# In[39]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lmodel = lm.fit(x_train, y_train)


# In[44]:


y_pred = lmodel.predict(x_test)
print('MSE :', mean_squared_error(y_test, y_pred))
print('RMSE :', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R2 score: ',r2_score(y_test,y_pred))


# In[ ]:


from sklearn


# In[41]:


from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score


# In[43]:


cv_score1 = cross_val_score(lmodel, x_train, y_train, cv=10, scoring='r2').mean()
cv_score2 = cross_val_score(lmodel, x_test, y_test, cv=10, scoring='r2').mean()
print(cv_score1)
print(cv_score2)


# In[ ]:





# In[ ]:





# # Gradient Boosting Algorithm

# In[56]:


from sklearn.ensemble import GradientBoostingRegressor
gbm = GradientBoostingRegressor()
gbm.fit(x_train, y_train)


# In[57]:


cv_score1 = cross_val_score(gbm, x_train, y_train, cv=10, scoring='r2').mean()
cv_score2 = cross_val_score(gbm, x_test, y_test, cv=10, scoring='r2').mean()
print(cv_score1)
print(cv_score2)


# In[ ]:


####


# # RandomForest Regressor

# In[58]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(x_train, y_train)


# In[59]:


cv_score1 = cross_val_score(rfr, x_train, y_train, cv=10, scoring='r2').mean()
cv_score2 = cross_val_score(rfr, x_test, y_test, cv=10, scoring='r2').mean()
print(cv_score1)
print(cv_score2)


# In[ ]:





# In[ ]:




