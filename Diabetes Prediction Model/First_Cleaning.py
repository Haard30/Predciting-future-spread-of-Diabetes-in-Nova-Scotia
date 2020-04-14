
# coding: utf-8

# In[36]:


import pandas as pd


# In[37]:


cols = ['Year','Zone','Sex','Agegroup','Population','Prevalence','CPR']
diabetes_data = pd.read_csv('Diabetes_Crude_Prevalence.csv',names=cols)


# In[38]:


diabetes_data.describe()
diabetes_data = diabetes_data[1:]


# In[39]:


New_genders = []

for gender in diabetes_data.Sex:
    if gender=="F":
        New_genders.append(5)
    elif gender=="M":
        New_genders.append(10)
    else:
        print(gender)
        


# In[40]:


diabetes_data.Sex = New_genders


# In[41]:


Min_ages = []
Max_ages = []

for age_range in diabetes_data.Agegroup:
    if "to" in age_range:
        min_age = int(age_range[:2])
        max_age = int(age_range[6:])
        Min_ages.append(min_age)
        Max_ages.append(max_age)
    if "+" in age_range:
        min_age = 80
        max_age = 109
        Min_ages.append(min_age)
        Max_ages.append(max_age)
        


# In[42]:


print(Min_ages[:5])


# In[43]:


print(Max_ages[:5])


# In[44]:


diabetes_data = diabetes_data.drop('Agegroup',axis=1)


# In[45]:


diabetes_data.insert(3,"Min_Age",Min_ages,True)


# In[46]:


diabetes_data.insert(4,"Max_Age",Max_ages,True)


# In[47]:


diabetes_data.to_csv('cleaned_diabetes_data.csv',encoding="UTF-8",header=True)


# In[48]:


diabetes_data.Prevalence = diabetes_data['Prevalence'].str.replace(',','').astype(int)


# In[49]:


diabetes_data.Population = diabetes_data['Population'].str.replace(',','').astype(int)


# In[50]:


diabetes_data.to_csv('final_cleaned_diabetes_data.csv',encoding="UTF-8",header=True)

