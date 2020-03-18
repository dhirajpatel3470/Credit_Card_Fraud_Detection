#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud Detection  

# ##This model is made to detect the fraudulent transaction.
# The datasets contains transactions made by credit cards in September 2013 by european cardholders.
# This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
# 
# It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

# In[47]:


### importing Necessary Librarie
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import seaborn as sns


# In[48]:


### load the dataset from the csv file using pandas
data=pd.read_csv(r'D:\creditcardfraud\creditcard.csv')


# In[49]:


data.head()


# In[50]:


### print the shape of data
print(data.shape)


# In[51]:


data.describe()


# In[52]:


### plot the histogram of of each parameter
data.hist(figsize=(20,20))
plt.show()


# In[53]:


### Determine the number of fraud cases in dataset
import seaborn as sns
Fraud=data[data['Class']==1]
Valid=data[data['Class']==0]
outlier_fraction=len(Fraud)/float(len(Valid))
print(outlier_fraction)
print('fraud cases: {}'.format(len(Fraud)))
print('Valid cases: {}'.format(len(Valid)))


# In[54]:


### correlation Matrix
corrmat=data.corr()
fig=plt.figure(figsize=(12,9))
sns.heatmap(corrmat,vmax=0.8,square=True)


# In[55]:


### Get all the columns from dataframe
columns=data.columns.tolist()
columns=[c for c in columns if c not in ["Class"]]
X=data[columns]
Y=data["Class"]
##print shapes
print(X.shape)
print(Y.shape)


# In[56]:


from  sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
##define outlier detection tools to be compared
classifiers = {
    "Isolation Forest":IsolationForest(n_estimators=100, max_samples=len(X), 
                                       contamination=outlier_fraction,random_state=1, verbose=0),
    "Local Outlier Factor":LocalOutlierFactor(n_neighbors=20, algorithm='auto', 
                                              leaf_size=30, metric='minkowski',
                                              p=2, metric_params=None, contamination=outlier_fraction)
   }


# In[57]:


### Fit the model
plt.figure(figsize=(9,7))
n_outliers=len(Fraud)
for i, (clf_name,clf) in enumerate(classifiers.items()):
    if clf_name=="Local Outlier Factor":
        y_pred=clf.fit_predict(X)
        score_pred=clf.negative_outlier_factor_
    else:
        clf.fit(X)
        score_pred=clf.decision_function(X)
        y_pred=clf.predict(X)
    # Reshape the prediction values to 0 for valid ,1 for fraud.
    y_pred[y_pred==1]=0
    y_pred[y_pred==-1]=1
    n_errors=(y_pred != Y).sum()
    #run classification metrics 
    print('{}:{}'.format(clf_name,n_errors))
    print(accuracy_score(Y,y_pred))
    print(" Classification report: ")
    print(classification_report(Y,y_pred))


# # Using CNN

# In[58]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense,Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D,MaxPool1D
from tensorflow.keras.optimizers import Adam


# In[59]:


### Balance dataset
Valid=Valid.sample(Fraud.shape[0])


# In[60]:


data=Fraud.append(Valid,ignore_index=True)
data.head()


# In[61]:


data['Class'].value_counts()


# In[62]:


X = data.drop('Class', axis = 1)
y = data['Class']


# In[63]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0,stratify=y)


# In[64]:


X_train.shape,X_test.shape


# In[65]:


scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


# In[66]:


y_train=y_train.to_numpy()
y_test=y_test.to_numpy()


# In[67]:


X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


# In[68]:


X_train.shape, X_test.shape


# In[69]:


model = Sequential()
model.add(Conv1D(32, 2, activation='relu', input_shape = X_train[0].shape))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Conv1D(64, 2, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))


# In[70]:


model.summary()


# In[71]:


model.compile(optimizer=Adam(lr=0.0001), loss = 'binary_crossentropy', metrics=['accuracy'])
epochs=20


# In[72]:


history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=1)


# In[73]:


import matplotlib.pyplot as plt
# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[74]:


model.save('credit_card.h5')


# In[ ]:




