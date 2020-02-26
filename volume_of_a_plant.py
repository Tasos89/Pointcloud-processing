# volume of a cluster made via region growing

#%% import libraries
from laspy.file import File
import pptk
import numpy as np
from math import radians, sin, cos, asin, sqrt, atan
from scipy.spatial import ConvexHull
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib

#%% import cluster data made with the use of "region growing" algorithm
path = r"C:\Users\laptop\Google Drive\scripts\Pointcloud-processing/region_growing_no_dense.las"
data_las = File(path, mode = 'r')

xyz = np.vstack([data_las.x, data_las.y, data_las.z]).transpose()
rgb = (np.c_[data_las.Red, data_las.Green, data_las.Blue])/255/255

# choose cluster by color
idx_sort = np.argsort(rgb[:,0])
sorted_records_array = rgb[idx_sort]
vals, idx_start, count =np.unique(sorted_records_array, return_counts=True,return_index=True)
res = np.split(idx_sort, idx_start[1:])

# choose the number of cluster 
xyz_2 = xyz[res[0]]

nbrs = NearestNeighbors(n_neighbors = 12, algorithm = 'kd_tree').fit(xyz_2)
distances, indices = nbrs.kneighbors(xyz_2)

xyz_2 = xyz_2[indices[1,:]]
#v = pptk.viewer(xyz_2)
#v.set(point_size=0.01)

# calculate the convex hull (volume of a cluster)
# function that calculates the distance among 2 points
def calc_dist_tuple(x1,y1,x2,y2):
    dx = x1 - x2
    dy = y1 - y2
    dist = sqrt(dx**2 + dy**2)
    phi = atan(dy/dx)
    return dx, dy    #dist/cos(phi), dist/sin(phi)

# creation local coordinate system
new_data = np.zeros(xyz_2.shape)
for i in range(xyz_2.shape[0]):
    new_data[i,:2] = calc_dist_tuple(xyz_2[0,0],xyz_2[0,1],xyz_2[i,0],xyz_2[i,1])
new_data[:,2] = xyz_2[:,2]

# convex hull
ch = ConvexHull(new_data)
print('The volume of the cluster is: ',ch.volume,'m^3')

#%% visulaize the points consist the convex hull
data = new_data
fig,ax = plt.subplots(figsize=(12, 8))
sctr = ax.scatter(x=data[:,0],y=data[:,1],c=data[:,2])
ax.set_title('broccoli plants points')
plt.colorbar(sctr, ax=ax, format='$%d')
plt.colorbar( ax=ax, format='$%d')
