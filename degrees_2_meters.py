# Convertion degrees to meters in local coordinate system

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