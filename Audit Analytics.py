#!/usr/bin/env python
# coding: utf-8

# ## Import Python packages
# 

# In[61]:


#work and manipulate with dataframe
import pandas as pd
import numpy as np


# In[62]:


#Graph visuals
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[63]:


#For iteration and other useful functions
import itertools


# In[64]:


#train and split data
from sklearn import model_selection


# In[65]:


#Logistic Regression algorithm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score, f1_score, precision_score,confusion_matrix, recall_score, roc_auc_score


# In[66]:


#Random forest algorithm
from sklearn.ensemble import RandomForestClassifier


# ## Data understanding and Manipulation
# 

# In[67]:


#let's fetch the data and convert into python datafram
df = pd.read_csv('audit_data.csv')


# In[68]:


#Peek into the data (First and last 5 rows)
df.head()
df.tail()


# In[69]:


#List all the columns available in the dataset
df.columns


# In[70]:


#After understanding the data I came down to a conclusion that columns 'TOTAL' and 'LOCATION_ID' will not be helpful for further analysis
#It's never a good practice to delet any columns , incase those columns helps in future
#so creating a new dataframe excluding those columns
df_audit = df[['Sector_score', 'PARA_A', 'Score_A', 'Risk_A', 'PARA_B',
       'Score_B', 'Risk_B', 'numbers', 'Score_B.1', 'Risk_C',
       'Money_Value', 'Score_MV', 'Risk_D', 'District_Loss', 'PROB', 'RiSk_E',
       'History', 'Prob', 'Risk_F', 'Score', 'Inherent_Risk', 'CONTROL_RISK',
       'Detection_Risk', 'Audit_Risk', 'Risk']]


# In[71]:


#understand the dataset
df_audit.info()

# 25 columns, 776 rows, data type either in float/integer


# In[72]:


#Understand the stats of the dataset
df_audit.describe()


# In[73]:


#Find any mssing/NAN/NAN values in the dataset
df_audit.isna().sum()


# In[74]:


#ONly one na value in the column of 'Money_Value'. 
#While working with the dataset to do Data Analysis and Data Modelling it is highly recommended to not keep any row or column containing missing/nan/na values
#Let's see which row is it, and what we can do with that row
rows_with_NAN = df_audit.isnull().any(axis=1)
display(df_audit[rows_with_NAN])


# In[75]:


#Since only one columne has na value. Also. Due to less number of the rows available and in order to prevent any loss of information, I will impute the missing value
#Imputation can be done in many ways based on the missing information. Here we could have imputed the Money_value by meaning/moding value of the company under same sector
#Particular firm is under which section not known, so meaning with all the values
df_audit['Money_Value'].fillna((df_audit['Money_Value'].mean()), inplace=True)


# In[76]:


#checking if the value is been imputed or not
df_audit.isna().sum()


# In[77]:


df_audit=df_audit.drop(['Detection_Risk'],axis=1)


# In[78]:


#keeping target variable seperate from the dataframne
X=df_audit.drop(['Risk'],axis=1)


# ## Data Correlation 

# In[79]:


X.corr(method='pearson').style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)


# In[80]:


#stats about target variable. To check how many firms are considered at risk and how many are not 
sns.countplot(df_audit['Risk'], label="Count")


# ## Data Modeling

# In[81]:


#Let's start working on building model
#Train Test split dataset
y = df_audit['Risk']

#import train_test_split and cross_val_score
from sklearn.model_selection import train_test_split,cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,stratify=y, random_state = 123)


# In[82]:


#Feature Scaling - to make sure all thevalues in the dataset are standardized with each other, so model can easily correlae and predict target values
#importing StandardScaler to standardized
from sklearn.preprocessing import StandardScaler
X_train_scaled = pd.DataFrame(StandardScaler().fit_transform(X_train))
X_test_scaled = pd.DataFrame(StandardScaler().fit_transform(X_test))


# In[83]:


logi = LogisticRegression(random_state = 0)
logi.fit(X_train_scaled, y_train)


# In[84]:


kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle= True)
scoring = 'accuracy'

acc_logi = cross_val_score(estimator = logi, X = X_train_scaled, y = y_train, cv = kfold,scoring=scoring)
acc_logi.mean()


# In[85]:


y_predict_logi = logi.predict(X_test_scaled)
acc= accuracy_score(y_test, y_predict_logi)
roc=roc_auc_score(y_test, y_predict_logi)
prec = precision_score(y_test, y_predict_logi)
rec = recall_score(y_test, y_predict_logi)
f1 = f1_score(y_test, y_predict_logi)

results = pd.DataFrame([['Logistic Regression',acc, acc_logi.mean(),prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy','Cross Val Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results


# In[86]:



random_forest_e = RandomForestClassifier(n_estimators = 100,criterion='entropy', random_state = 47)
random_forest_e.fit(X_train_scaled, y_train)


# In[87]:


acc_rande = cross_val_score(estimator = random_forest_e, X = X_train_scaled, y = y_train, cv = kfold, scoring=scoring)
acc_rande.mean()


# ## Performance Evaluation

# In[88]:


y_predict_r = random_forest_e.predict(X_test_scaled)
roc=roc_auc_score(y_test, y_predict_r)
acc = accuracy_score(y_test, y_predict_r)
prec = precision_score(y_test, y_predict_r)
rec = recall_score(y_test, y_predict_r)
f1 = f1_score(y_test, y_predict_r)

model_results = pd.DataFrame([['Random Forest',acc, acc_rande.mean(),prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy','Cross Val Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results = results.append(model_results, ignore_index = True)
results


# In[89]:


cm_logi = confusion_matrix(y_test, y_predict_logi)
plt.title('Confusion matrix of the Logistic classifier')
sns.heatmap(cm_logi,annot=True,fmt="d")
plt.show()


# In[90]:


cm_r = confusion_matrix(y_test, y_predict_r)
plt.title('Confusion matrix of the Random Forest classifier')
sns.heatmap(cm_r,annot=True,fmt="d")
plt.show()


# ## Feature Significance for further analysis by experts

# In[59]:


importances = random_forest_e.feature_importances_
indices = np.argsort(importances)[::-1]

# Rearrange feature names so they match the sorted feature importances
names = [X.columns[i] for i in indices]

# Create plot
plt.figure()

# Create plot title
plt.title("Feature Importance")

# Add bars
plt.bar(range(X.shape[1]), importances[indices])

# Add feature names as x-axis labels
plt.xticks(range(X.shape[1]), names, rotation=90)

# Show plot
plt.show()


# In[ ]:




