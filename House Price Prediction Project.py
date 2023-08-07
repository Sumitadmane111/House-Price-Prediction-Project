#!/usr/bin/env python
# coding: utf-8

# # Read CSV file

# In[54]:


import pandas as pd


# In[55]:


from warnings import filterwarnings
filterwarnings("ignore")


# In[56]:


df=pd.read_csv("C:\\Users\\shree kalika\\OneDrive\\Desktop\\deta science\\ethilive\\python,machine learning\\files\\project machine learning\\training_set.csv")
df


# # EDA 

# In[57]:


df.info()


# In[58]:


df.describe()


# # Check missing values

# In[63]:


df.isna().sum()


# # Define x and y

# In[62]:


x=df.drop(['SalePrice'],axis=1)
y=df['SalePrice']


# # Sklearn Pipeline

# In[13]:


cat=[]
con=[]

for i in x.columns:
    if x[i].dtypes=='object':
        cat.append(i)
    else:
        con.append(i)


# In[14]:


cat


# In[15]:


con


# In[16]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer


# In[17]:


num_pipe=Pipeline(steps=([('impute',SimpleImputer(strategy='mean')),('scaler',StandardScaler())]))
cat_pipe=Pipeline(steps=([('impute',SimpleImputer(strategy='most_frequent')),('encode',OrdinalEncoder())]))
preprocess1=ColumnTransformer([('num_pipe',num_pipe,con),('cat_pipe',cat_pipe,cat)])


# In[18]:


x1=pd.DataFrame(preprocess1.fit_transform(x),columns=x.columns)
x1


# # Feature selection

# In[82]:


from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
sfs=SequentialFeatureSelector(lr,n_features_to_select=70)
sfs.fit(x1,y)


# In[83]:


cols=sfs.get_feature_names_out()
cols


# In[84]:


x2=pd.DataFrame(sfs.fit_transform(x1,y),columns=cols)
x2


# In[85]:


x_final=pd.DataFrame(x,columns=cols)
x_final


# # Sklearn Pipeline

# In[86]:


cat2=[]
con1=[]

for i in x_final.columns:
    if x_final[i].dtypes=='object':
        cat2.append(i)
    else:
        con1.append(i)


# In[87]:


con1


# In[88]:


cat2


# In[89]:


num_pipe1=Pipeline(steps=([('impute1',SimpleImputer(strategy='mean')),('scaler1',StandardScaler())]))
cat_pipe1=Pipeline(steps=([('impute1',SimpleImputer(strategy='most_frequent')),('encode1',OneHotEncoder())]))
pre1=ColumnTransformer([('num1',num_pipe1,con1),('cat1',cat_pipe1,cat2)])


# In[90]:


x_new=pre1.fit_transform(x_final)
x_new


# In[91]:


cols1=pre1.get_feature_names_out()
cols1


# In[92]:


x_new=pd.DataFrame(x_new,columns=cols1)
x_new


# # Train Test Split

# In[93]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_new,y,test_size=0.2,random_state=21)
lr.fit(x_train,y_train)


# # MSE, RMSE, MAE, R2_score

# In[94]:


from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import numpy as np


# In[95]:


y_pred_train=lr.predict(x_train)
print('Training Data Evaluation')
print('*'*50)
MSE=mean_squared_error(y_pred_train,y_train)
print('MSE:',MSE)

RMSE=np.sqrt(MSE)
print(RMSE)

MAE=mean_absolute_error(y_pred_train,y_train)
print('MAE:',MAE)

R2=r2_score(y_pred_train,y_train)
print('R2',R2)


# In[96]:


y_pred=lr.predict(x_test)
print('Testing Data Evaluation')
print('*'*50)

MSE1=mean_squared_error(y_pred,y_test)
print('MSE:',MSE1)

RMSE1=np.sqrt(MSE1)
print(RMSE1)

MAE1=mean_absolute_error(y_pred,y_test)
print('MAE:',MAE1)

R_2=r2_score(y_pred,y_test)
print('R2',R_2)


# # Lasso
# 

# In[36]:


from sklearn.linear_model import Lasso,Ridge
la=Lasso(alpha=1)
la.fit(x_train,y_train)


# In[37]:


from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import numpy as np

y_pred_train=la.predict(x_train)
print('Training Data Evaluation')
print('*'*50)

MSE=mean_squared_error(y_pred_train,y_train)
print('MSE:',MSE)

RMSE=np.sqrt(MSE)
print('RMSE:',RMSE)

MAE=mean_absolute_error(y_pred_train,y_train)
print('MAE:',MAE)

R2=r2_score(y_pred_train,y_train)
print('R2_score:',R2)


# In[38]:


y_pred=la.predict(x_test)

print('Testing Data Evaluation')
print('*'*50)
MSE=mean_squared_error(y_pred,y_test)
print('MSE:',MSE)

RMSE=np.sqrt(MSE)
print('RMSE:',RMSE)

MAE=mean_absolute_error(y_pred,y_test)
print('MAE:',MAE)

R2=r2_score(y_pred,y_test)
print('R2_score:',R2)


# # Ridge

# In[40]:


ra=Ridge()
ra.fit(x_train,y_train)


# In[41]:


y_pred_train=ra.predict(x_train)
print('Training Data Evaluation')
print('*'*50)

MSE=mean_squared_error(y_pred_train,y_train)
print('MSE:',MSE)

RMSE=np.sqrt(MSE)
print('RMSE:',RMSE)

MAE=mean_absolute_error(y_pred_train,y_train)
print('MAE:',MAE)

R2=r2_score(y_pred_train,y_train)
print('R2_score:',R2)


# In[42]:


y_pred=ra.predict(x_test)

print('Testing Data Evaluation')
print('*'*50)
MSE=mean_squared_error(y_pred,y_test)
print('MSE:',MSE)

RMSE=np.sqrt(MSE)
print('RMSE:',RMSE)

MAE=mean_absolute_error(y_pred,y_test)
print('MAE:',MAE)

R2=r2_score(y_pred,y_test)
print('R2_score:',R2)


# # Model Predict the test.csv 

# In[44]:


df2=pd.read_csv("C:\\Users\\shree kalika\\OneDrive\\Desktop\\deta science\\ethilive\\python,machine learning\\files\\project machine learning\\testing_set.csv")
df2


# In[45]:


x_sample=pre1.transform(df2)
x_sample


# In[46]:


x_samp=pd.DataFrame(x_sample,columns=cols1)
x_samp


# In[47]:


y_pred1=lr.predict(x_samp)
y_pred1


# In[51]:


prediction=df2[['Id']]
prediction


# In[50]:


prediction['SalePrice']=y_pred1
prediction


# # Download file

# In[101]:


prediction.to_csv('result of project',index=False)

