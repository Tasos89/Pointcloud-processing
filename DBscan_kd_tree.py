#%% DBscanner
# I nice implementation of DBscaner 
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs

data = pd.read_csv(r'C:\Users\laptop\Google Drive\scripts\Pointcloud-processing\out_file.csv')
#data = pd.read_csv(r'C:\Users\laptop\Google Drive\scripts\Pointcloud-processing\dense_cloud.csv')

data = np.asarray(data)
x = np.array(data[:,0])
y = np.array(data[:,1])
z = np.array(data[:,2])

xn = (x - x.min()) / (x.max() - x.min())
yn = (y - y.min()) / (y.max() - y.min())
zn = (z - z.min()) / (z.max() - z.min())
xyz_nn = np.vstack([xn,yn,zn]).T



db = DBSCAN(eps=0.02, min_samples=10).fit(xyz_nn)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

# #############################################################################
# Plot result
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(data[:,0],data[:,1],data[:,2])
plt.show()

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = data[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=0.1)

    xy = data[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=1)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

#%% K-means
# How to obtain the optimal number of clusters
# https://blog.cambridgespark.com/how-to-determine-the-optimal-number-of-clusters-for-k-means-clustering-14f27070048f
import sklearn
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import pandas as pd
import pptk

data = pd.read_csv(r'C:\Users\laptop\Google Drive\scripts\Pointcloud-processing\out_file.csv')
#data = pd.read_csv(r'C:\Users\laptop\Google Drive\scripts\Pointcloud-processing\dense_cloud.csv')

data = np.asarray(data)
x = np.array(data[:,0])
y = np.array(data[:,1])
z = np.array(data[:,2])

xn = (x - x.min()) / (x.max() - x.min())
yn = (y - y.min()) / (y.max() - y.min())
zn = (z - z.min()) / (z.max() - z.min())
xyz_nn = np.vstack([xn,yn,zn]).T


#K-Means implementation
model = KMeans(240)
model.fit(data)
print(model.cluster_centers_)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(data[:,0],data[:,1],data[:,2])
plt.show()



plt.scatter(data[:,0],data[:,1],linewidths=0.01);

plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], s=100, color="red"); # Show the centres

# chooce the number of clusters
Sum_of_squared_distances = []
K = range(1,200)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data)
    Sum_of_squared_distances.append(km.inertia_)


plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


