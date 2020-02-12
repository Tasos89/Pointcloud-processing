#%% DBSCAN
# First nice implementation of DBscaner 
# https://stackoverflow.com/questions/53076159/dbscan-silhouette-coefficients-does-this-for-loop-work
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
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

db = DBSCAN(eps=0.02, min_samples=5).fit(xyz_nn) #0.02 has been found later via elbow..
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

#%% DBSCAN
# How to obtain the optimal number of clusters
# https://blog.cambridgespark.com/how-to-determine-the-optimal-number-of-clusters-for-k-means-clustering-14f27070048f
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import pptk
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneebow.rotor import Rotor

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

#1rst evaluation
# Nearest neighbors to find the optimal epsilon (maximum distance) https://towardsdatascience.com/machine-learning-clustering-dbscan-determine-the-optimal-value-for-epsilon-eps-python-example-3100091cfbc
nbrs = NearestNeighbors(n_neighbors = 5, algorithm = 'kd_tree').fit(xyz_nn) #['auto', 'ball_tree', 'kd_tree', 'brute']
distances, indices = nbrs.kneighbors(xyz_nn) #the indices of the nearest neighbors 
distances = np.sort(distances, axis=0)
distances = distances[:,4]
plt.plot(distances)

y = np.array(distances)
x = np.linspace(0,len(x),len(x))
xy = np.vstack((x,y)).T

rotor = Rotor()
rotor.fit_rotate(xy)
elbow_idx = rotor.get_elbow_index()
rotor.plot_elbow()
eps = distances[elbow_idx]/2
del x,y,xy

clustering = DBSCAN( algorithm = 'kd_tree',eps=eps, min_samples=5).fit(xyz_nn) #the number of samples is D+1=4
labels = clustering.labels_

colors = [int(i % 23) for i in labels] # 554 labels to 23 distinguished colors

v = pptk.viewer(data,colors)
v.set(point_size=0.01)

# matplotlib
core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
core_samples_mask[clustering.core_sample_indices_] = True
labels = clustering.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

# #############################################################################
# Plot result
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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


# convert data to lon,lat
from math import radians, sin, cos, asin, sqrt

data[:,0] = data[:,0]*4/(10**5)
data[:,1] = data[:,1]/(10**4)

def calc_dist_tuple(lat1,lon1,lat2,lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (sin(dlat / 2) ** 2, cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2)
    a1 = a[0]; a2 = a[1]
    c1 = 2 * asin(sqrt(a1))
    m1 = 1000 * 6371 * c1
    c2 = 2 * asin(sqrt(a2))
    m2 = 1000 * 6371 * c2
    return m1,m2

new_data = np.zeros(data.shape)
for i in range(data.shape[0]):
    new_data[i,:2] = calc_dist_tuple(data[0,1],data[0,0],data[i,1],data[i,0])
new_data[:,2] = data[:,2]

# =============================================================================
# indices = []
# for i in range(np.max(labels)):
#     indices.append(np.where(labels==i))
# =============================================================================
    
## Test of one random cluster ##    
indices = [np.where(labels==i) for i in range(np.max(labels))]

xyz_300 = new_data[indices[300]]
v = pptk.viewer(xyz_300)
v.set(point_size=0.002)

# convex hull of a cluster

from scipy.spatial import ConvexHull
ch = ConvexHull(xyz_300)
simplices = np.column_stack((np.repeat(ch.vertices[0], ch.nsimplex),
                                 ch.simplices))

def tetrahedron_volume(a, b, c, d):
    return np.abs(np.einsum('ij,ij->i', a-d, np.cross(b-d, c-d))) / 6

tets = ch.points[simplices]
np.sum(tetrahedron_volume(tets[:, 0], tets[:, 1],
                                     tets[:, 2], tets[:, 3]))



# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html    ###############
# https://stackoverflow.com/questions/24733185/volume-of-convex-hull-with-qhull-from-scipy
# scipy delaunay
#is the way to extract the pogol bounding box from a cluster of points
# a different way is to to find the min max etc
# =============================================================================
# =============================================================================
# # import startin
# # dt = startin.DT()
# # dt.insert(data)
# # dt.write_obj(r"C:\Users\laptop\Google Drive\scripts\Pointcloud-processing\triangle_data.obj")
# =============================================================================
# =============================================================================

# find the concave hull of each cluster-broccoli
# https://gist.github.com/dwyerk/10561690
# https://plot.ly/python/alpha-shapes/


### 2nd method to calculate a suitable value for epsilon ####
# Its does not perform well
import sklearn
n=10
scores = np.zeros(n)
epslist = np.linspace(eps+eps/10,eps-eps/10,n)
for i in range(n):
    print(i)
    epsi = epslist[i]
    
    clustering = DBSCAN(eps=epsi, min_samples=5).fit(xyz_nn)
    labels = clustering.labels_
    scores[i] = sklearn.metrics.silhouette_score(xyz_nn, labels, metric='euclidean')
#evaluate the eps looking at the histogram
#plt.plot(epslist,scores)

ind_max = np.argmax(scores)
eps = epslist[ind_max] #the optimal distance after Shilouette evaluetion

clustering = DBSCAN(eps, min_samples=5).fit(xyz_nn) #the number of samples is D+1=4
labels = clustering.labels_

colors = [int(i % 23) for i in labels]

v = pptk.viewer(data,colors)
v.set(point_size=0.01)

#%% K-Means Shilouette evaluation (Works)
# https://towardsdatascience.com/clustering-metrics-better-than-the-elbow-method-6926e1f723a6
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd


print(__doc__)

data = pd.read_csv(r'C:\Users\laptop\Google Drive\scripts\Pointcloud-processing\out_file.csv')
data = np.asarray(data)
x = np.array(data[:,0])
y = np.array(data[:,1])
X = np.vstack((x,y)).T


#range_n_clusters = [300,320, 400]
range_n_clusters = [700]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='.',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

plt.show()


#%% Region growing segmentation 
# https://github.com/davidcaron/pclpy/issues/9
# it works for the no_dense point cloud
# region growing
import math
import pclpy
from pclpy import pcl
from laspy.file import File
import pptk
import numpy as np

pc = pclpy.io.las.read( r"C:\Users\laptop\Google Drive\scripts\Pointcloud-processing\no_dense_cloud.las", "PointXYZRGBA")
rg = pcl.segmentation.RegionGrowing.PointXYZRGBA_Normal()
rg.setInputCloud(pc)
normals_estimation = pcl.features.NormalEstimationOMP.PointXYZRGBA_Normal()
normals_estimation.setInputCloud(pc)
normals = pcl.PointCloud.Normal()
normals_estimation.setRadiusSearch(0.01)  #no_dense = 0.01
normals_estimation.compute(normals)
rg.setInputNormals(normals)

rg.setMaxClusterSize(700) #no_dense 700
rg.setMinClusterSize(8) #no_dense 8
rg.setNumberOfNeighbours(4) #no_dense 4
rg.setSmoothnessThreshold(5 / 180 * math.pi)
rg.setCurvatureThreshold(30)
rg.setResidualThreshold(10)
clusters = pcl.vectors.PointIndices()
rg.extract(clusters)
cloud = rg.getColoredCloud()
#pclpy.io.las.write(cloud, r"C:\Users\laptop\Google Drive\scripts\Pointcloud-processing/region_growing_no_dense.las")
cloud.show()

path = r"C:\Users\laptop\Google Drive\scripts\Pointcloud-processing/region_growing_no_dense.las"
data_las = File(path, mode = 'r')
#extract the coordinates of the .las file
xyz = np.vstack([data_las.x, data_las.y, data_las.z]).transpose()
rgb = (np.c_[data_las.Red, data_las.Green, data_las.Blue])/255/255

# http://www.pcl-users.org/Finding-oriented-bounding-box-of-a-cloud-td4024616i20.html

# discriminate the clusters of the xyz based on color
idx_sort = np.argsort(rgb[:,0])
sorted_records_array = rgb[idx_sort]
vals, idx_start, count =np.unique(sorted_records_array, return_counts=True,return_index=True)

# sets of indices
res = np.split(idx_sort, idx_start[1:])

##filter them with respect to their size, keeping only items occurring more than once
#vals = vals[count > 1]
#res = filter(lambda x: x.size > 1, res)

xyz_2 = xyz[res[108]]
v = pptk.viewer(xyz_2)
v.set(point_size=0.02)

import startin

dt = startin.DT()
dt.insert(xyz_2)
triangles = np.asarray(dt.all_triangles())
vertices = np.asarray(dt.all_vertices())
vertices = vertices[1:,:]

# remove triangles with long edges
threshold = 0.07
remove = []
for i in range(len(triangles)):
    tr = triangles[i,:]
    a = tr[0]
    b = tr[1]
    c = tr[2]
    a_cor = vertices[a-1,:]
    b_cor = vertices[b-1,:]
    c_cor = vertices[c-1,:]
    dis_a_b = np.sqrt(np.square(a_cor[0]-b_cor[0])+np.square(a_cor[1]-b_cor[1]))
    dis_b_c = np.sqrt(np.square(c_cor[0]-b_cor[0])+np.square(c_cor[1]-b_cor[1]))
    dis_a_c = np.sqrt(np.square(c_cor[0]-a_cor[0])+np.square(c_cor[1]-a_cor[1]))
    check_a = [dis_a_b,dis_a_c]
    if dis_a_b>threshold and dis_a_c>threshold:
        remove.append(a)
    if dis_a_b>threshold and dis_b_c>threshold:
        remove.append(b)
    if dis_a_c>threshold and dis_b_c>threshold:
        remove.append(c)
    

vertices = np.delete(vertices,remove,axis=0)

v = pptk.viewer(vertices)
v.set(point_size=0.01)

