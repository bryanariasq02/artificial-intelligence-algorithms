from matplotlib import markers, transforms
import pandas as pd
import numpy as np
import sklearn
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#Clustering of dataset with sklearn KMeans
dataset = pd.read_csv(r'./docs/credit_train.csv')
dataset["Term"] = dataset["Term"].replace("Short Term", "0") #Transform in INT
dataset["Term"] = dataset["Term"].replace("Long Term", "1") #Transform in INT
dataset["Term"] = dataset["Term"].astype(np.float16) 
dataset = np.array(dataset)
dataset = dataset[:,4:7] #Select Term, Credit score and Annual Income 
#print(dataset[:10])

#clean data, delete values with "na"
dataset = dataset[pd.notnull(dataset[:,0])] #Filter without NAN in term
dataset = dataset[pd.notnull(dataset[:,1])] #Filter without NAN in Credit score
dataset = dataset[pd.notnull(dataset[:,2])] #Filter without NAN in Annual Income
print(dataset[:10])

#Shortening the data to 200 rows
dataset = dataset[:1000]

X_train = np.array(dataset)

#Data scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

Modelo_Kmeans = KMeans (n_clusters=4)
Modelo_Kmeans.fit(X_train)
Centroides = Modelo_Kmeans.cluster_centers_
Y_train = Modelo_Kmeans.labels_
print('\n\tCentroides encontrados')
print (Centroides)
print('')

print('Mostrando datos y clase asignada')
for i in range(0,len(X_train)): 
  print (X_train[i],' ',Y_train[i])
print('')

color_theme = np.array(['c', 'r', 'y', 'g'])
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X_train[:,0],X_train[:,1], X_train[:,2],  c=color_theme[Y_train])
ax.set_xlabel('Term, Short/Long')
ax.set_ylabel('Credit Score')
ax.set_zlabel('Annual Income')

plt.show()

#Predict

Target=np.ones((1, 3))
Target[0,0]=1 #Term - Long Term
Target[0,1]=600 #Credit Score
Target[0,2]=420000 #Annuela icome

print(Target.shape)
#Making predictions
print (Modelo_Kmeans.predict (Target))