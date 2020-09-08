import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



labels = pd.read_csv('info.csv' , header=None)
n_females = len(labels[labels[1] == 'female'])
n_males = len(labels[labels[1] == 'male'])
genders = pd.DataFrame(labels[1]).to_numpy()

labels = labels.rename(columns={1: 'gender'})
sns.set(style="darkgrid") 

f = labels[labels['gender'] == 'female']
m = labels[labels['gender'] == 'male']

plt.hist(labels[2] , bins = 45 )
plt.xlabel('year of birth')
plt.ylabel('count')

plt.hist([f[2] , m[2]] , bins = 30 , label = ['female' , 'male'])
plt.xlabel('year of birth')
plt.ylabel('count')
plt.legend()


ax = sns.countplot(x='gender', data=labels)