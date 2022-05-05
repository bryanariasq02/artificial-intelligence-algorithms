import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from Clasificador import *

#Fetal_health
dataset = pd.read_csv(r'docs/fetal_health.csv')
# print(dataset.shape)
# print(dataset.info())
# print(dataset.isnull().sum())
dataset = np.array(dataset)

#1. Dividing dataset into input (X) and output (Y) variables
X = dataset[:,0:21]
Y = dataset[:,21]

# print("X")
# print(X[:10])
# print("Y")
# print(Y[:10])

# Split input and output into training and testing sets
X_train, X_test,Y_train, Y_test= train_test_split(X,Y,test_size=0.33,random_state=14541)

# Data normalization

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# print("X_train")
# print(X_train[:10])
# print("Y_train")
# print(Y_train[:10])

KNN(X_train, Y_train, X_test, Y_test)




