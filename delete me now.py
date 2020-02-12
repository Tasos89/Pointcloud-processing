import math
from laspy.file import File
import pptk
import numpy as np


path = r"C:\Users\laptop\Google Drive\scripts\Pointcloud-processing/region_growing_no_dense.las"
data_las = File(path, mode = 'r')
#extract the coordinates of the .las file
xyz = np.vstack([data_las.x, data_las.y, data_las.z]).transpose()
rgb = (np.c_[data_las.Red, data_las.Green, data_las.Blue])/255/255


idx_sort = np.argsort(rgb[:,0])
sorted_records_array = rgb[idx_sort]
vals, idx_start, count =np.unique(sorted_records_array, return_counts=True,return_index=True)
res = np.split(idx_sort, idx_start[1:])

xyz_2 = xyz[res[0]]
v = pptk.viewer(xyz_2)
v.set(point_size=0.02)

from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors = 12, algorithm = 'kd_tree').fit(xyz_2) 
distances, indices = nbrs.kneighbors(xyz_2) 

xyz_2 = xyz_2[indices[1,:]]
v = pptk.viewer(xyz_2)
v.set(point_size=0.01)

########## calculate the convex hull (volume of a cluster)
from math import radians, sin, cos, asin, sqrt

xyz_2[:,0] = xyz_2[:,0]*4/(10**5)
xyz_2[:,1] = xyz_2[:,1]/(10**4)
xyz_2[:,2] = -xyz_2[:,2]



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

new_data = np.zeros(xyz_2.shape)
for i in range(xyz_2.shape[0]):
    new_data[i,:2] = calc_dist_tuple(xyz_2[0,1],xyz_2[0,0],xyz_2[i,1],xyz_2[i,0])
new_data[:,2] = xyz_2[:,2]

from scipy.spatial import ConvexHull
ch = ConvexHull(new_data)
simplices = np.column_stack((np.repeat(ch.vertices[0], ch.nsimplex),
                                 ch.simplices))

def tetrahedron_volume(a, b, c, d):
    return np.abs(np.einsum('ij,ij->i', a-d, np.cross(b-d, c-d))) / 6

tets = ch.points[simplices]
np.sum(tetrahedron_volume(tets[:, 0], tets[:, 1],
                                     tets[:, 2], tets[:, 3]))

# =============================================================================
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.tri as mtri
# from scipy.spatial import Delaunay
# 
# tri = Delaunay(xyz_2)
# 
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# ax.plot_trisurf(xyz_2[:,0], xyz_2[:,1], xyz_2[:,2], triangles=tri.simplices, cmap=plt.cm.Spectral)
# plt.show()
# =============================================================================




import startin
dt = startin.DT()
dt.insert(xyz_2)
triangles = np.asarray(dt.all_triangles())
vertices = np.asarray(dt.all_vertices())
vertices = vertices[1:,:]

threshold = 0.08
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

del dt,triangles
dt = startin.DT()
dt.insert(vertices)

b = dt.write_obj(r"C:\Users\laptop\Google Drive\cluster_tri.obj")
