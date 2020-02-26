# from degrees to local coordinate system in meters and calculate the convex hull of those 

from scipy.spatial import ConvexHull
from math import radians, sin, cos, asin, sqrt

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

ch = ConvexHull(new_data)
#ch.volume
simplices = np.column_stack((np.repeat(ch.vertices[0], ch.nsimplex),
                                 ch.simplices))

def tetrahedron_volume(a, b, c, d):
    return np.abs(np.einsum('ij,ij->i', a-d, np.cross(b-d, c-d))) / 6


tets = ch.points[simplices]
np.sum(tetrahedron_volume(tets[:, 0], tets[:, 1],
                                     tets[:, 2], tets[:, 3]))

 