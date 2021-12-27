#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Link github: 
#Nama: Drajat Danu Wardana
#Nim: 41819010009
#Linear Regression
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
advertising = pd.read_csv("Company_data.csv")
advertising


# In[2]:



advertising.shape
advertising.info()
advertising.describe()


# In[3]:



import matplotlib.pyplot as plt 
import seaborn as sns

sns.pairplot(advertising, x_vars=['TV', 'Radio','Newspaper'], 
             y_vars='Sales', size=4, aspect=1, kind='scatter')
plt.show()


# In[4]:


sns.heatmap(advertising.corr(), cmap="YlGnBu", annot = True)
plt.show()


# In[5]:



X = advertising['TV']
y = advertising['Sales']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, 
                                                    test_size = 0.3, random_state = 100)
X_train
y_train


# In[6]:


import matplotlib.pyplot as plt 
import seaborn as sns
plt.scatter(X_train, y_train)
plt.plot(X_train, 6.948 + 0.054*X_train, 'r')
plt.show()


# In[35]:


# Importing Statsmodels.api library from Stamodel package
import statsmodels.api as sm

# Adding a constant to get an intercept
X_train_sm = sm.add_constant(X_train)


# In[36]:


# Fitting the resgression line using 'OLS'
lr = sm.OLS(y_train, X_train_sm).fit()

# Printing the parameters
lr.params


# In[38]:


# Adding a constant to X_test
X_test_sm = sm.add_constant(X_test)

# Predicting the y values corresponding to X_test_sm
y_test_pred = lr.predict(X_test_sm)

# Printing the first 15 predicted values
y_test_pred


# In[39]:



from sklearn.metrics import r2_score

r_squared = r2_score(y_test, y_test_pred)
r_squared

plt.scatter(X_test, y_test)
plt.plot(X_test, y_test_pred, 'r')
plt.show()


# In[6]:


#Logistic Regression
#Nama: Drajat Danu Wardana
#Nim: 41819010009
#Logistic Regression
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# In[4]:


uas2 = pd.read_csv("testcpns.csv")
uas2


# In[3]:


dataFrame.describe()


# In[13]:


sns.pairplot(uas2, x_vars=['ipk', 'pengalaman_kerja','toefl'],
            y_vars='diterima', size=4, aspect=1, kind='scatter')
plt.show()


# In[18]:


#import libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# In[19]:


#load dataset
dataFrame = pd.read_csv("testcpns.csv")
dataFrame.head(5)


# In[20]:


#check the data if there's a NaN value
dataFrame.isna().values.any()

#check the features
print(dataFrame.dtypes)


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#train the model
model = LogisticRegression()
model.fit(X_train, y_train)


# In[23]:


y_pred = model.predict(X_test)
print(y_pred)


# In[24]:


from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[25]:


import seaborn as sns
import numpy as np
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
Text(0.5, 257.44, 'Predicted label')


# In[26]:


#check precision, recall, f1-score
print(classification_report(y_test, y_pred))
print("accuracy: ", accuracy_score(y_test, y_pred))


# In[27]:


# Importing naumpy and pandas libraries to read the data

# Supress Warnings
import warnings
warnings.filterwarnings('ignore')

# Import the numpy and pandas package
import numpy as np
import pandas as pd

# Read the given CSV file, and view some sample records
uas = pd.read_csv("testcpns.csv")
uas


# In[28]:


sns.scatterplot(x="ipk",y="toefl",data =uas2, hue="diterima", size=5)
plt.show()


# In[ ]:




