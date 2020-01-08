
def knn(xyz,k):
    from sklearn.neighbors import NearestNeighbors #import NearestNeigbhors package
    nbrs = NearestNeighbors(n_neighbors = k, algorithm = 'auto').fit(xyz) #['auto', 'ball_tree', 'kd_tree', 'brute']
    distances, indices = nbrs.kneighbors(xyz) #the indices of the nearest neighbors
    return indices 