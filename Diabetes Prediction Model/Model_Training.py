
# coding: utf-8

# In[11]:



import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import numpy as np


# In[3]:


cols = ['Year','Zone','Sex','Min_Age','Max_Age','Population','Prevalence','CPR']
mydata = pd.read_csv('final_cleaned_diabetes_data.csv',names=cols)
mydata = mydata[1:]


# In[4]:


np.set_printoptions(formatter={'float': lambda x: "{0:0.13f}".format(x)})
X = mydata.iloc[:,:6].values.astype(float)
Y = mydata.iloc[:,6].values.astype(float)



X_5 = []
for x in X:
    X_5.append(x[5])


# In[5]:


print(X_5[:5])


# In[6]:


scaler = StandardScaler()
X_5 = preprocessing.scale(X_5)

print(X_5[:5])


# In[7]:


for i in range(X.__len__()):
    X[i][5] = X_5[i]


# In[8]:


print(X[:5])


# In[9]:


X_norm = preprocessing.normalize(X)


# In[10]:


print(X_norm[:5])


# In[41]:


from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LeakyReLU,Dropout
from keras.optimizers import Adam,RMSprop
from keras.activations import relu

model = Sequential()
model.add(Dense(25, input_dim=6, activation='tanh'))
model.add(Dense(45, activation='relu'))
model.add(Dense(65, activation='relu'))
model.add(Dense(85, activation='relu'))
model.add(Dense(105, activation='relu'))
model.add(Dense(105, activation='relu'))
model.add(Dense(85, activation='relu'))
model.add(Dense(85, activation='relu'))
model.add(Dense(65, activation='relu'))
model.add(Dense(45, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1,activation='linear'))
model.compile(loss='mean_absolute_percentage_error', optimizer=Adam(lr=0.0001))
model.fit(X_norm,Y, epochs=1000,validation_split=0.15,batch_size=10)

model.save('Diabetes.model')
model.save_weights('Diabetes_Weights.h5')


# In[111]:


ip_vector = [2020,5,5,20,29,17180]

for i in range(ip_vector.__len__()):
    ip_vector[i] = float(ip_vector[i])

X = mydata.iloc[:,:6].values.astype(float)

lx = list(X)
lx.append(ip_vector)
X = np.array(lx)


# In[112]:


X_5 = []
for x in X:
    X_5.append(x[5])

X_5 = preprocessing.scale(X_5)


for i in range(X.__len__()):
    X[i][5] = X_5[i]

X_norm = preprocessing.normalize(X)


# In[113]:


print(X_norm.shape)
predicted_values = model.predict(X_norm)

print(predicted_values[-1])


# In[118]:


def compute(ip_vector):

    for i in range(ip_vector.__len__()):
        ip_vector[i] = float(ip_vector[i])

    X = mydata.iloc[:,:6].values.astype(float)

    lx = list(X)
    lx.append(ip_vector)
    X = np.array(lx)

    X_5 = []
    for x in X:
        X_5.append(x[5])

    X_5 = preprocessing.scale(X_5)


    for i in range(X.__len__()):
        X[i][5] = X_5[i]

    X_norm = preprocessing.normalize(X)

    predicted_values = model.predict(X_norm)

    print(predicted_values[-1][0])
    return predicted_values[-1][0]


# In[128]:


pred_data = pd.read_csv('prediction.csv')

Predicted_values = []

for i in pred_data.values:
    ans = compute(i)
    Predicted_values.append(int(ans))
    
    


# In[137]:


Ratios = []

i = 0
for value in Predicted_values:
    temp = value / pred_data.values[i][5]
    i+=1
    Ratios.append(temp*100)


# In[142]:


Zones = []

for i in pred_data.values:
    if i[1]==5:
        Zones.append("Central")
    elif i[1]==15:
        Zones.append("Eastern")
    elif i[1]==10:
        Zones.append("Northern")
    elif i[1]==20:
        Zones.append("Western")
    else:
        Zones.append("ERROR")


# In[144]:


Population = []

for i in pred_data.values:
    Population.append(i[5])


# In[148]:


final_list = []

for i in range(len(Population)):
    temp = []
    temp.append(Zones[i])
    temp.append(Predicted_values[i])
    temp.append(Population[i])
    temp.append(Ratios[i])
    final_list.append(temp)
    
cols = ['Zone','Predicted_Diabetes_Count','Total_Population','Ratio']
final_df = pd.DataFrame(final_list,columns=cols)

final_df.to_csv('FINAL.csv',encoding="UTF-8")

