from sklearn.cluster import KMeans 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Kmeans Sklearn

#Import data
url = 'docs/Mall_Customers.csv'
df = pd.read_csv(url)
print(df)

#Clean data (Just "Ingreso" and "Puntuación")
df = df.iloc[:, [3,4]].values
print(df)

##colors for plot after the clustering
color_theme = np.array(['grey', 'b', 'y', 'g', 'c', 'violet'])

#Graph for total Mall customers
plt.subplot(2,2,1)
plt.plot(df[:,0], df[:,1], 'ob')
plt.xlabel('Ingreso')
plt.ylabel('Puntuación')
plt.title('Mall customers')


#Creating the model by KMeans
clustering = KMeans(n_clusters=4, random_state=5).fit(df)
labels = clustering.labels_  #Output of labels 0,1,2,3,4
predict = clustering.predict([[90, 50], [12, 3]]) #Predict the cluster to which it belongs
centroids = clustering.cluster_centers_

print("\n\tSets\n",labels, "\n\n\t Predicts of  [[90, 50], [12, 3]]\n", predict, "\n\n\t Centroids\n", centroids)

#Plot for centroids from MKmeas
plt.subplot(2,2,2)
plt.xlabel('Ingreso')
plt.ylabel('Puntuación')
plt.title('Centroids')
plt.plot(centroids[:,0], centroids[:,1], 'or')

#Plot centroids with data
plt.subplot(2,2,3)
plt.plot(df[:,0], df[:,1], 'ob')
plt.xlabel('Ingreso')
plt.ylabel('Puntuación')
plt.title('Centroids inside data')
plt.plot(centroids[:,0], centroids[:,1], 'or')

#Plot with different colors by clostering

#Plot centroids with data
plt.subplot(2,2,4)
plt.scatter(x = df[:,0], y = df[:,1], c=color_theme[clustering.labels_], s=50)
plt.xlabel('Ingreso')
plt.ylabel('Puntuación')
plt.title('K-Means classification')
plt.plot(centroids[:,0], centroids[:,1], '*r')

plt.legend()
plt.show()