#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
train=pd.read_csv('../../Downloads/train.csv')
test=pd.read_csv('../../Downloads/test.csv')
ideal=pd.read_csv('ideal.csv')


# In[4]:


print(train.head())


# In[5]:


print(test.head())


# In[8]:


train.shape,test.shape,ideal.shape


# In[5]:


import pandas as pd
import numpy as np

train=pd.read_csv('../../Downloads/train.csv')
test=pd.read_csv('../../Downloads/test.csv')
ideal=pd.read_csv('ideal.csv')

train.head()
test.head()
ideal.head()

train.shape,test.shape,ideal.shape

train.columns

#Mean Squared Error

li=[] #index value will be stored here
least_squares=[]
for j in range(1,len(train.columns)):
  lse_sq=[]
  for k in range(1,len(ideal.columns)):
    mse_val=0
    for l in range(len(train)):
      t1=train.iloc[l,j]
      i1=ideal.iloc[l,k]
      mse_val=mse_val + ((t1-i1)**2)
    lse_sq.append(mse_val/len(train))
  min_least = min(lse_sq)
  index = lse_sq.index(min_least)
  li.append(index+1)
  least_squares.append(min_least)

lse_sq

len(lse_sq)

task_1 = pd.DataFrame({"Column_Name":li, "LSE":least_squares})
task_1

ideal

ideals = []
for i in range(0,4):
  ideals.append(ideal[['x',f'y{str(task_1.iloc[i,0])}']])
ideals

for i in ideals:
  test_1 = test.merge(i,on='x',how='left')
test_1

ideal_index = []
deviation = []
for a in range(len(test_1)):  #rows
  mse_li = []
  for b in range(2,len(test_1.columns)):  #columns
    z1 = test_1.iloc[a,1]
    z2 = test_1.iloc[a,1]
    mse = ((z1-z2)**2)
    mse_li.append(mse)
  #print(mse_li)
  #print(min(mse_li))
  min_least = min(mse_li)
  if min_least<np.sqrt(2)*0.085616:
    deviation.append(min_least)
    index = mse_li.index(min_least)
    ideal_index.append(index)
  else:
    deviation.append(min_least)
    ideal_index.aappend('miss')

test_1['Deviation'] = deviation
test_1['Ideal Index'] = ideal_index

test_1





# In[ ]:




