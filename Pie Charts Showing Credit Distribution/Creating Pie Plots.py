
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


cols = ['Year','Payment_type','subcategory','Amount']
phy_data = pd.read_csv('Medical_Payments_to_Physicians_in_Nova_Scotia.csv',names=cols)
phy_data = phy_data[1:]


# In[12]:


data_2012_13 = []
for row in phy_data.values:
    if row[0]=="2018-19":
        data_2012_13.append(row)

dt_2012_13 = pd.DataFrame(data_2012_13,columns=cols)
dt_2012_13.to_csv('Data_18_19.csv',encoding="utf-8")


# In[119]:


data_2012_13 = []

pd_d_12_13 = pd.read_csv('Data_15_16.csv',names=cols,header=None)
pd_d_12_13 = pd_d_12_13[1:]
print(pd_d_12_13[:5])
pd_d_12_13.Amount = pd_d_12_13['Amount'].str.replace(',','').astype(float)


# In[120]:


pTypes_unique = set(pd_d_12_13.Payment_type)


# In[121]:


A_sums = []

for pType in pTypes_unique:
    sum1=0
    for row in pd_d_12_13.values:
        if row[1]==pType:
            sum1+=row[3]
           
            
    
    A_sums.append(sum1)

pTypes = list(pTypes_unique)
print(pTypes)


# In[122]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
ax.pie(A_sums, labels = pTypes,autopct='%1.2f%%')
plt.savefig('tessstttyyy.png', dpi=100)
plt.show()
plt.close(fig)

