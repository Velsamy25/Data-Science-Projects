#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing required python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[2]:


# first method to load the dataset
#loan_pred_df=pd.read_csv(r"C:\Vel\Self-Learning\DataSets\Loan Prediction\train_ctrUa4K.csv")
# second method to load the data set
import io
get_ipython().magic(u'cd "C:\\Vel\\Self-Learning\\DataSets\\Loan Prediction"')
loan_pred_df=pd.read_csv("train_ctrUa4K.csv")


# In[3]:


#Viewing 1st 5 records of data set
loan_pred_df.head()


# In[4]:


# checking observations and variables 
loan_pred_df.shape


# In[5]:


# checking the available columns
loan_pred_df.columns


# In[6]:


# checking the data types of each variables and missing values
loan_pred_df.info()


# In[7]:


# Checking the missing values
missing_values=loan_pred_df.isna().sum().sort_values(ascending=False)


# In[8]:


missing_values


# In[9]:


loan_pred_df.Credit_History.value_counts()


# In[10]:


# Statistical inference 
loan_pred_df.describe()


# In[22]:


#columns in the data set
loan_pred_df.columns


# In[32]:


# Checking the outlier of loan amount using boxplot
loan_pred_df.boxplot(column='LoanAmount')


# In[34]:


# Checking the outlier of applicant income using boxplot
loan_pred_df.boxplot(column='ApplicantIncome')


# In[35]:


loan_pred_df.boxplot(column='CoapplicantIncome')


# In[39]:


# Checking the relationship between graduation and loan approval status using cross tab
pd.crosstab(loan_pred_df['Education'],loan_pred_df['Loan_Status'],margins=True)


# In[43]:


# Checking the relationship between credit history and loan approval status using cross tab
pd.crosstab(loan_pred_df['Credit_History'],loan_pred_df['Loan_Status'],margins=True)


# In[146]:


#Creating a new variable "Total Income" using "Applicant income and Coapplicant income"
loan_pred_df['Total Income']=loan_pred_df['ApplicantIncome']+loan_pred_df['CoapplicantIncome']


# In[51]:


# Checking applicant income
loan_pred_df.hist(column='ApplicantIncome',bins=10)


# In[53]:


# Checking Loan Amount income
loan_pred_df.hist(column='LoanAmount',bins=10)


# In[139]:


loan_pred_df.hist(column='CoapplicantIncome',bins=10)


# In[147]:


#Checking total income
loan_pred_df['Total Income'].hist(bins=20)


# In[148]:


# Removing the outliers using numpy log function to distribute equally
loan_pred_df['LoanAmount_log']=np.log(loan_pred_df['LoanAmount'])
loan_pred_df['ApplicantIncome_log']=np.log(loan_pred_df['ApplicantIncome'])
loan_pred_df['Total Income_log']=np.log(loan_pred_df['Total Income'])


# In[141]:


# now loan amount is adjusted correctly post applying log function
loan_pred_df['LoanAmount_log'].hist(bins=10)


# In[142]:


loan_pred_df.ApplicantIncome_log.hist(bins=10)


# In[149]:


loan_pred_df['Total Income_log'].hist(bins=10)


# In[68]:


# Filling missing categorical variables with mode and numerical values with mean
loan_pred_df['Credit_History'].fillna(loan_pred_df['Credit_History'].mode()[0],inplace=True)
loan_pred_df['Self_Employed'].fillna(loan_pred_df['Self_Employed'].mode()[0],inplace=True)
loan_pred_df['LoanAmount'].fillna(loan_pred_df['LoanAmount'].mean(),inplace=True)
loan_pred_df['Dependents'].fillna(loan_pred_df['Dependents'].mode()[0],inplace=True)
loan_pred_df['Loan_Amount_Term'].fillna(loan_pred_df['Loan_Amount_Term'].mode()[0],inplace=True)
loan_pred_df['Gender'].fillna(loan_pred_df['Gender'].mode()[0],inplace=True)
loan_pred_df['Married'].fillna(loan_pred_df['Married'].mode()[0],inplace=True)
loan_pred_df['LoanAmount_log'].fillna(loan_pred_df['LoanAmount_log'].mean(),inplace=True)


# In[150]:


# missing values are imputed
loan_pred_df.isna().sum()


# In[161]:


# dropping the columns which are not required 
loan_pred_df.drop(['LoanAmount','ApplicantIncome','CoapplicantIncome','Total Income'],axis=1,inplace=True)


# In[164]:


loan_pred_df.drop(['Loan_ID','ApplicantIncome_log','CoapplicantIncome_log'],axis=True,inplace=True)


# In[166]:


loan_pred_df.shape


# In[167]:


# Splitting the based on data types
categorical_cols=loan_pred_df.select_dtypes(include=np.object)
numerical_cols=loan_pred_df.select_dtypes(include=np.number)
print(categorical_cols.shape)
print(numerical_cols.shape)


# In[168]:


categorical_cols.info()


# In[169]:


numerical_cols.info()


# In[170]:


categorical_cols.head()


# In[171]:


# converting the categorical data into numbers using label encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[172]:


categorical_cols_new=categorical_cols.apply(le.fit_transform)


# In[173]:


# Categorical values are changed to numbers
categorical_cols_new.head()


# In[174]:


# Combining both categorical and numerical data sets together
main_df=pd.concat([numerical_cols,categorical_cols_new],axis=1)


# In[175]:


main_df.head()


# In[176]:


# Creating dependent and independent variables
x=main_df.drop(['Loan_Status'],axis=1)
y=main_df['Loan_Status']
print(main_df.shape)
print(x.shape)
print(y.shape)


# In[179]:


# Creating train and test data 
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)


# In[180]:


print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)


# In[184]:


# Applying machine learning models
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(criterion='entropy',random_state=0)
model=dtc.fit(xtrain,ytrain)


# In[188]:


y_predict=model.predict(xtest)
y_predict


# In[189]:


# Calculating the accuracy score to understand how model performs
# DecisionTreeClassifier scores seems to be very less and this can be exlored with diff algorithms
from sklearn import metrics
metrics.accuracy_score(y_predict,ytest)


# In[190]:


# Checking if Naive Bayes algorithm fits for this model
from sklearn.naive_bayes import GaussianNB
NBClassifier=GaussianNB()


# In[204]:


model=NBClassifier.fit(xtrain,ytrain)


# In[205]:


y_pred=model.predict(xtest)


# In[206]:


y_pred


# In[222]:


# Looks like Naive Bayes gives better score
accuracy_score = metrics.accuracy_score(y_pred,ytest)
accuracy_score


# In[223]:


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,classification_report


# In[224]:


confusion_matrix(ytest,y_pred)


# In[225]:


TP,FP,FN,TN=confusion_matrix(ytest,y_pred).ravel()
TP,FP,FN,TN


# In[226]:


# formula for F1_score is 2*Precision*Recall/(Recall+Precision)
f1_score=metrics.f1_score(ytest,y_pred)
f1_score


# In[227]:


#Precision score = TP/TP+FP
precision_score=metrics.precision_score(y_pred,ytest)
precision_score

