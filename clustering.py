import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# data preprocessing
data = pd.read_csv('voices.csv')
df = data
df = df.drop(['gender'],axis = 1)
df = df.drop(['age'],axis = 1)
X = df.values
y = data['gender'].values
encoder = LabelEncoder()
y = encoder.fit_transform(y)
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# fitting clustering algorithm on data
n_clstrs = [2 , 50 , 290]
predicted_labels = np.zeros((len(X) , len(n_clstrs)))
for i in range(len(n_clstrs)):
    c = n_clstrs[i]
    clstr = GaussianMixture(n_components = c)
    predicted_labels[:,i] = clstr.fit_predict(X)

# presenting clustered groups
groups = []
for i in range(len(n_clstrs)):
    m = n_clstrs[i]
    a = []
    for j in range(m):
        a.append(data[predicted_labels[:,i] == j])
    groups.append(a)

    

