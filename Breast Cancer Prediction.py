#!/usr/bin/env python
# coding: utf-8

# # Breast Cancer Prediciton

# In[1]:


import numpy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[2]:


df=pd.read_csv("data.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.isna().sum()


# In[6]:


df.shape


# In[7]:


df=df.dropna(axis=1)


# In[8]:


df.shape


# In[9]:


df.describe()


# In[10]:


df['diagnosis'].value_counts()


# In[11]:


sns.countplot(df['diagnosis'],label="count")


# In[12]:


from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
df.iloc[:,1]=labelencoder_Y.fit_transform(df.iloc[:,1].values)


# In[13]:


df.head()


# In[14]:


sns.pairplot(df.iloc[:,1:5],hue="diagnosis")


# In[15]:


df.iloc[:,1:32].corr()


# In[16]:


plt.figure(figsize=(10,10))
sns.heatmap(df.iloc[:,1:10].corr(),annot=True,fmt=".0%")


# In[17]:


X=df.iloc[:,2:31].values
Y=df.iloc[:,1].values


# In[18]:


print(Y)


# In[19]:


print(X)


# In[20]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)


# In[21]:


from sklearn.preprocessing import StandardScaler
X_train=StandardScaler().fit_transform(X_train)
X_test=StandardScaler().fit_transform(X_test)


# In[22]:


#models

def models(X_train,Y_train):
    #logistic regression
    from sklearn.linear_model import LogisticRegression
    log=LogisticRegression(random_state=0)
    log.fit(X_train,Y_train)
    
    
    #Decision Tree
    from sklearn.tree import DecisionTreeClassifier
    tree=DecisionTreeClassifier(random_state=0,criterion="entropy")
    tree.fit(X_train,Y_train)
    
    
    #Random Forest
    from sklearn.ensemble import RandomForestClassifier
    forest=RandomForestClassifier(random_state=0,criterion="entropy",n_estimators=10)
    forest.fit(X_train,Y_train)
    
    print('[0]logistic regression accuracy:', log.score(X_train,Y_train))
    print('[1]Decision tree accuracy:', tree.score(X_train,Y_train))
    print('[2]Random forest accuracy:', forest.score(X_train,Y_train))
    
    
    return log,tree,forest


# In[23]:


model=models(X_train,Y_train)


# In[24]:


# testing the models

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

for i in range(len(model)):
    print("Model",i)
    print(classification_report(Y_test,model[i].predict(X_test)))
    print('Accuracy :',accuracy_score(Y_test,model[i].predict(X_test)))


# In[25]:


pred=model[2].predict(X_test)
print('Predicted Values')
print(pred)
print('Actual Values')
print(Y_test)


# In[ ]:




