# Import packages

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import pptk
from laspy.file import File
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from pyntcloud import PyntCloud
import startin
from sklearn.cluster import DBSCAN

#%% broccoli label=1

# Define the path of the trainning data
input_Broccoli = r"C:\Users\laptop\Google Drive\Shared folder Tasos-VanBoven\Sample_data\Broccoli\classes\broccoli3.las"

# read the trainning data as .las file         
data_las = File(input_Broccoli, mode='r') 

# Extract the coordinates and rgb values fro .las file
xyz_B = np.vstack([data_las.x, data_las.y, data_las.z]).transpose()
rgb_B = ((np.c_[data_las.Red, data_las.Green, data_las.Blue]) / 255.) / 255. #normalized
R = rgb_B[:,0]
G = rgb_B[:,1]
B = rgb_B[:,2]

# Normalize the xyz
xn = (data_las.x - data_las.x.min()) / (data_las.x.max() - data_las.x.min())
yn = (data_las.y - data_las.y.min()) / (data_las.y.max() - data_las.y.min())
zn = (data_las.z - data_las.z.min()) / (data_las.z.max() - data_las.z.min())
xyz_nn = np.vstack([xn,yn,zn]).T

# Nearest neighbors
nbrs = NearestNeighbors(n_neighbors = 35, algorithm = 'kd_tree').fit(xyz_nn) #['auto', 'ball_tree', 'kd_tree', 'brute']
distances, indices_B = nbrs.kneighbors(xyz_nn) #the indices of the nearest neighbors 

# Extract geometrical features
linearity = []
planarity = []
scatter = []
omnivariance = []
anisotropy = []
#eigenentropy = []
change_curvature = []
dif_elev = []
mean_elev = []
omnivariance = []

for i in range(len(indices_B)):
    ind = indices_B[i]
    #coords = xyz_B[(ind),:]
    coords = xyz_nn[(ind),:]
    x = coords[:,0]
    y = coords[:,1]
    z = coords[:,2]

    data_new = np.vstack((x,y,z)) #without normalize data
    cov_matrix = np.cov(data_new)
    e ,v = np.linalg.eig(cov_matrix)
    e_sorted = np.sort(e)
    e = e_sorted[::-1] #λ1>λ2>λ3>0

    omni = (e[0]*e[1]*e[2])**(1/3)
    omnivariance.append(omni)
    lin = (e[0]-e[1])/e[0]
    linearity.append(lin)
    plan = (e[1]-e[2])/e[0]
    planarity.append(plan)
    sc = e[2]/e[0]
    scatter.append(sc)
    anis = (e[0]-e[2])/e[0]
    anisotropy.append(anis)
    cha = e[2]/sum(e)
    change_curvature.append(cha)
    m_el = z.mean()
    mean_elev.append(m_el)
    d_el = z.max()-z.min()
    dif_elev.append(d_el)
    
omnivariance = np.asarray(omnivariance)
l = np.asarray(linearity)
p = np.asarray(planarity)
s = np.asarray(scatter)
an = np.asarray(anisotropy)
ch = np.asarray(change_curvature)
m_e = np.asarray(mean_elev)
d_e = np.asarray(dif_elev)

# normalization of the geometrical features
omn_n_B = (omnivariance -omnivariance.min()) / (omnivariance.max() - omnivariance.min())
lin_n_B = (l -l.min()) / (l.max() - l.min())
plan_n_B = (p -p.min()) / (p.max() - p.min())
scat_n_B = (s -s.min()) / (s.max() - s.min())
an_n_B = (an -an.min()) / (an.max() - an.min())
ch_cur_n_B = (ch -ch.min()) / (ch.max() - ch.min())
mean_el_n_B = (m_e -m_e.min()) / (m_e.max() - m_e.min())
dif_elev_n_B = (d_e -d_e.min()) / (d_e.max() - d_e.min())

# from BGR to Lab # https://gist.github.com/bikz05/6fd21c812ef6ebac66e1
# =============================================================================
# def func(t):
#     if (t > 0.008856):
#         return np.power(t, 1/3.0);
#     else:
#         return 7.787 * t + 16 / 116.0;
# matrix = [[0.412453, 0.357580, 0.180423],
#           [0.212671, 0.715160, 0.072169],
#           [0.019334, 0.119193, 0.950227]]
# Lab_OpenCv = []
# Lab_B = np.zeros((len(rgb_B),3))
# for row in rgb_B:
#     cie = np.dot(matrix, row);
#     cie[0] = cie[0] /0.950456;
#     cie[2] = cie[2] /1.088754; 
#     L = 116 * np.power(cie[1], 1/3.0) - 16.0 if cie[1] > 0.008856 else 903.3 * cie[1];
#     a = 500*(func(cie[0]) - func(cie[1]));
#     b = 200*(func(cie[1]) - func(cie[2]));
#     Lab_B = [b , a, L]; 
#     L = L * 255 / 100;
#     a = a + 128;
#     b = b + 128;
#     Lab_OpenCv.append([b,a,L])
# Lab_OpenCv = np.asarray(Lab_OpenCv)
# scaler = MinMaxScaler()
# Lab_B = scaler.fit_transform(Lab_OpenCv)
# Lab_B[:,2] = 0.0
# b_B = Lab_B[:,0]
# a_B = Lab_B[:,1]
# L_B = Lab_B[:,2]
# =============================================================================

Broccoli_LABEL = 1.*np.ones(len(xyz_B))
if 'Lab_B' in locals():
    features_B = np.vstack((omn_n_B, dif_elev_n_B, L_B, a_B, b_B, Broccoli_LABEL)).T #Lab
else:
    features_B = np.vstack((omn_n_B, dif_elev_n_B, R, G ,B, Broccoli_LABEL)).T #rgb

#%% Grass label=2

# Define the path of the trainning data
input_Grass = r"C:\Users\laptop\Google Drive\Shared folder Tasos-VanBoven\Sample_data\Broccoli\classes\grass3.las"

# read the trainning data as .las file        
data_las = File(input_Grass, mode='r') 

# Extract the coordinates and rgb values fro .las file
xyz_G = np.vstack([data_las.x, data_las.y, data_las.z]).transpose()
rgb_G = ((np.c_[data_las.Red, data_las.Green, data_las.Blue]) / 255.) / 255. #normalized
R = rgb_G[:,0]
G = rgb_G[:,1]
B = rgb_G[:,2]

#normalize the xyz
xn = (data_las.x - data_las.x.min()) / (data_las.x.max() - data_las.x.min())
yn = (data_las.y - data_las.y.min()) / (data_las.y.max() - data_las.y.min())
zn = (data_las.z - data_las.z.min()) / (data_las.z.max() - data_las.z.min())
xyz_nn = np.vstack([xn,yn,zn]).T

# Nearest neighbors
# =============================================================================
# nbrs = NearestNeighbors(n_neighbors = 35, algorithm = 'kd_tree').fit(xyz_std_G) #['auto', 'ball_tree', 'kd_tree', 'brute']
# distances, indices_G = nbrs.kneighbors(xyz_std_G) #the indices of the nearest neighbors 
# =============================================================================

nbrs = NearestNeighbors(n_neighbors = 35, algorithm = 'kd_tree').fit(xyz_nn) #['auto', 'ball_tree', 'kd_tree', 'brute']
distances, indices_G = nbrs.kneighbors(xyz_nn) #the indices of the nearest neighbors 

# Exract geometrical features
linearity = []
planarity = []
scatter = []
omnivariance = []
anisotropy = []
change_curvature = []
dif_elev = []
mean_elev = []
omnivariance = []

for i in range(len(indices_G)):
    ind = indices_G[i]
    #coords = xyz_G[(ind),:]
    coords = xyz_nn[(ind),:]
    x = coords[:,0]
    y = coords[:,1]
    z = coords[:,2]

    data_new = np.vstack((x,y,z)) #without normalize data
    cov_matrix = np.cov(data_new)
    e ,v = np.linalg.eig(cov_matrix)
    e_sorted = np.sort(e)
    e = e_sorted[::-1] #λ1>λ2>λ3>0

    omni = (e[0]*e[1]*e[2])**(1/3)
    omnivariance.append(omni)
    lin = (e[0]-e[1])/e[0]
    linearity.append(lin)
    plan = (e[1]-e[2])/e[0]
    planarity.append(plan)
    sc = e[2]/e[0]
    scatter.append(sc)
    anis = (e[0]-e[2])/e[0]
    anisotropy.append(anis)
    cha = e[2]/sum(e)
    change_curvature.append(cha)
    m_el = z.mean()
    mean_elev.append(m_el)
    d_el = z.max()-z.min()
    dif_elev.append(d_el)
    
omnivariance = np.asarray(omnivariance)
l = np.asarray(linearity)
p = np.asarray(planarity)
s = np.asarray(scatter)
an = np.asarray(anisotropy)
ch = np.asarray(change_curvature)
m_e = np.asarray(mean_elev)
d_e = np.asarray(dif_elev)

#normalization of the geometrical features
omn_n_G = (omnivariance -omnivariance.min()) / (omnivariance.max() - omnivariance.min())
lin_n_G = (l -l.min()) / (l.max() - l.min())
plan_n_G = (p -p.min()) / (p.max() - p.min())
scat_n_G = (s -s.min()) / (s.max() - s.min())
an_n_G = (an -an.min()) / (an.max() - an.min())
ch_cur_n_G = (ch -ch.min()) / (ch.max() - ch.min())
mean_el_n_G = (m_e -m_e.min()) / (m_e.max() - m_e.min())
dif_elev_n_G = (d_e -d_e.min()) / (d_e.max() - d_e.min())

# from BGR to Lab # https://gist.github.com/bikz05/6fd21c812ef6ebac66e1
# =============================================================================
# def func(t):
#     if (t > 0.008856):
#         return np.power(t, 1/3.0);
#     else:
#         return 7.787 * t + 16 / 116.0;
# matrix = [[0.412453, 0.357580, 0.180423],
#           [0.212671, 0.715160, 0.072169],
#           [0.019334, 0.119193, 0.950227]]
# Lab_OpenCv = []
# Lab_G = np.zeros((len(rgb_G),3))
# for row in rgb_G:
#     cie = np.dot(matrix, row);
#     cie[0] = cie[0] /0.950456;
#     cie[2] = cie[2] /1.088754; 
#     L = 116 * np.power(cie[1], 1/3.0) - 16.0 if cie[1] > 0.008856 else 903.3 * cie[1];
#     a = 500*(func(cie[0]) - func(cie[1]));
#     b = 200*(func(cie[1]) - func(cie[2]));
#     Lab_B = [b , a, L]; 
#     L = L * 255 / 100;
#     a = a + 128;
#     b = b + 128;
#     Lab_OpenCv.append([b,a,L])
# Lab_OpenCv = np.asarray(Lab_OpenCv)
# scaler = MinMaxScaler()
# Lab_G = scaler.fit_transform(Lab_OpenCv)
# Lab_G[:,2] = 0.0
# b_G = Lab_G[:,0]
# a_G = Lab_G[:,1]
# L_G = Lab_G[:,2]
# =============================================================================

Grass_LABEL = 2.*np.ones(len(xyz_G))
if 'Lab_G' in locals():
    features_G = np.vstack((omn_n_G, dif_elev_n_G, L_G, a_G, b_G, Grass_LABEL)).T # Lab
else:
    features_G = np.vstack((omn_n_G, dif_elev_n_G, R, G, B, Grass_LABEL)).T

#%% Soil label=3

# Define the path of the trainning data
input_Soil = r"C:\Users\laptop\Google Drive\Shared folder Tasos-VanBoven\Sample_data\Broccoli\classes\soil4.las"
      
# read the trainning data as .las file
data_las = File(input_Soil, mode='r') 

# Extract the coordinates and rgb values fro .las file
xyz_S = np.vstack([data_las.x, data_las.y, data_las.z]).T
#rgb_S = ((np.c_[data_las.Red, data_las.Green, data_las.Blue]) / 255.) / 255. #normalized
rgb_S = ((np.vstack((data_las.Red, data_las.Green, data_las.Blue)).T)/255)/255
R = rgb_S[:,0]
G = rgb_S[:,1]
B = rgb_S[:,2]

#normalize the xyz
xn = (data_las.x - data_las.x.min()) / (data_las.x.max() - data_las.x.min())
yn = (data_las.y - data_las.y.min()) / (data_las.y.max() - data_las.y.min())
zn = (data_las.z - data_las.z.min()) / (data_las.z.max() - data_las.z.min())
xyz_nn = np.vstack([xn,yn,zn]).T

# Nearest neighbors
# =============================================================================
# nbrs = NearestNeighbors(n_neighbors = 35, algorithm = 'kd_tree').fit(xyz_std_S) #['auto', 'ball_tree', 'kd_tree', 'brute']
# distances, indices_S = nbrs.kneighbors(xyz_std_S) #the indices of the nearest neighbors 
# =============================================================================

nbrs = NearestNeighbors(n_neighbors = 35, algorithm = 'kd_tree').fit(xyz_nn) #['auto', 'ball_tree', 'kd_tree', 'brute']
distances, indices_S = nbrs.kneighbors(xyz_nn) #the indices of the nearest neighbors 

# Extract geometrical features
linearity = []
planarity = []
scatter = []
omnivariance = []
anisotropy = []
change_curvature = []
dif_elev = []
mean_elev = []
omnivariance = []

for i in range(len(indices_S)):
    ind = indices_S[i]
    #coords = xyz_S[(ind),:]
    coords = xyz_nn[(ind),:]
    x = coords[:,0]
    y = coords[:,1]
    z = coords[:,2]

    data_new = np.vstack((x,y,z)) #without normalize data
    cov_matrix = np.cov(data_new)
    e ,v = np.linalg.eig(cov_matrix)
    e_sorted = np.sort(e)
    e = e_sorted[::-1] #λ1>λ2>λ3>0

    omni = (e[0]*e[1]*e[2])**(1/3)
    omnivariance.append(omni)
    lin = (e[0]-e[1])/e[0]
    linearity.append(lin)
    plan = (e[1]-e[2])/e[0]
    planarity.append(plan)
    sc = e[2]/e[0]
    scatter.append(sc)
    anis = (e[0]-e[2])/e[0]
    anisotropy.append(anis)
    cha = e[2]/sum(e)
    change_curvature.append(cha)
    m_el = z.mean()
    mean_elev.append(m_el)
    d_el = z.max()-z.min()
    dif_elev.append(d_el)
    
omnivariance = np.asarray(omnivariance)
l = np.asarray(linearity)
p = np.asarray(planarity)
s = np.asarray(scatter)
an = np.asarray(anisotropy)
ch = np.asarray(change_curvature)
m_e = np.asarray(mean_elev)
d_e = np.asarray(dif_elev)

#normalization of the data 
omn_n_S = (omnivariance -omnivariance.min()) / (omnivariance.max() - omnivariance.min())
lin_n_S = (l -l.min()) / (l.max() - l.min())
plan_n_S = (p -p.min()) / (p.max() - p.min())
scat_n_S = (s -s.min()) / (s.max() - s.min())
an_n_S = (an -an.min()) / (an.max() - an.min())
ch_cur_n_S = (ch -ch.min()) / (ch.max() - ch.min())
mean_el_n_S = (m_e -m_e.min()) / (m_e.max() - m_e.min())
dif_elev_n_S = (d_e -d_e.min()) / (d_e.max() - d_e.min())

# from BGR to Lab # https://gist.github.com/bikz05/6fd21c812ef6ebac66e1
# =============================================================================
# def func(t):
#     if (t > 0.008856):
#         return np.power(t, 1/3.0);
#     else:
#         return 7.787 * t + 16 / 116.0;
# matrix = [[0.412453, 0.357580, 0.180423],
#           [0.212671, 0.715160, 0.072169],
#           [0.019334, 0.119193, 0.950227]]
# Lab_OpenCv = []
# Lab_S = np.zeros((len(rgb_S),3))
# for row in rgb_S:
#     cie = np.dot(matrix, row);
#     cie[0] = cie[0] /0.950456;
#     cie[2] = cie[2] /1.088754; 
#     L = 116 * np.power(cie[1], 1/3.0) - 16.0 if cie[1] > 0.008856 else 903.3 * cie[1];
#     a = 500*(func(cie[0]) - func(cie[1]));
#     b = 200*(func(cie[1]) - func(cie[2]));
#     Lab_B = [b , a, L]; 
#     L = L * 255 / 100;
#     a = a + 128;
#     b = b + 128;
#     Lab_OpenCv.append([b,a,L])
# Lab_OpenCv = np.asarray(Lab_OpenCv)
# scaler = MinMaxScaler()
# Lab_S = scaler.fit_transform(Lab_OpenCv)
# Lab_S[:,2] = 0.0
# b_S = Lab_S[:,0]
# a_S = Lab_S[:,1]
# L_S = Lab_S[:,2]
# =============================================================================

Soil_LABEL = 3.*np.ones(len(xyz_S))
if 'Lab_S' in locals():
    features_S = np.vstack((omn_n_S, dif_elev_n_S, L_S, a_S, b_S, Soil_LABEL)).T # Lab
else:
    features_S = np.vstack((omn_n_S, dif_elev_n_S, R, G, B, Soil_LABEL)).T



#%% Train the classifier of Decision Tree
# https://towardsdatascience.com/decision-tree-in-python-b433ae57fb93
# https://www.datacamp.com/community/tutorials/random-forests-classifier-python
# https://towardsdatascience.com/discretisation-using-decision-trees-21910483fa4b

if 'Lab' in locals():
    feature_names = ['omnivariance','difference in elevation','L','a','b','label'] #Lab
else:
    if np.shape(features)[1] == 5:
        feature_names = ['omn','dif_elev','R','G','B','label'] #rgb
    elif np.shape(features)[1] == 11:
        feature_names = ['omnivariance','linearity','planarity','scattering','anisotropy','chan_curvature','mean_elev','dif_elev','R','G','B','label'] #all features
    elif np.shape(features)[1] == 10:
        feature_names = ['omnivariance','linearity','planarity','scattering','anisotropy','chan_curvature','dif_elev','R','G','B','label'] #except "mean_elevation"
        

labels = ['Broccoli','Grass','Soil'] # we have water as well

X = np.vstack((features_B,features_G,features_S))
X = pd.DataFrame(X, columns=feature_names)
y = pd.DataFrame(0, index=np.arange(len(X)),columns=labels)
y['Broccoli'][X['label']==1]=1
y['Grass'][X['label']==2]=1
y['Soil'][X['label']==3]=1
X = X.drop('label',axis=1) 

#split the training data into training and test data
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1)
#create the classifier
clr = DecisionTreeClassifier()
#train the classifier
clr.fit(x_train, y_train)

y_pred = clr.predict(x_test)
#confision matrix
species = np.array(y_test).argmax(axis=1)
predictions = np.array(y_pred).argmax(axis=1)
confusion_matrix(species, predictions) 

#%% Apply the classification using the trained classifier "Decision Tree" to the input .las data

feature_names = feature_names[:-1] # delete the column "labels"

xx = pd.DataFrame(features, columns=feature_names)

yy_pred = clr.predict(xx)

v = pptk.viewer(xyz,yy_pred)

#%% Train the classifier "Random forest"
# https://blog.goodaudience.com/machine-learning-using-decision-trees-and-random-forests-in-python-with-code-e50f6e14e19f
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(x_train,y_train)
yyy_pred = rfc.predict(x_test) #the predictions
#confussion matrix 
predictions_forest = np.array(yyy_pred).argmax(axis=1)
confusion_matrix(species,predictions_forest)

#%% classification with random forest
#https://www.geeksforgeeks.org/how-to-start-learning-machine-learning/

yyyy_pred = rfc.predict(xx)
v = pptk.viewer(xyz,yyyy_pred)
v.set(point_size=0.01)

#%%
# keep only the broccoli points by keeping the red points
red_indices = np.where(yyyy_pred[:,0]==1)
red_indices = np.reshape(red_indices,-1,1)
only_broccoli = xyz[red_indices,:]
rgb_broccoli = rgb[red_indices,:]

#visualization
v = pptk.viewer(only_broccoli,rgb_broccoli) 
v.set(point_size=0.01)

#%% voxelization 
# https://github.com/daavoo/pyntcloud 
# https://pyntcloud.readthedocs.io/en/latest/points.html
# https://medium.com/@shakasom/how-to-convert-latitude-longtitude-columns-in-csv-to-geometry-column-using-python-4219d2106dea
# https://github.com/mcoder2014/voxelization  !!!!!!!!!!!!!!!!!!

header = ['x','y','z']
only_broccoli = pd.DataFrame(data=only_broccoli, columns=header , index=None)
header_rgb = ['red','green','blue']
rgb_broccoli = pd.DataFrame(data=rgb_broccoli, columns=header_rgb , index=None)

XX = pd.DataFrame.join(only_broccoli, rgb_broccoli)

XX.to_csv(r'C:\Users\laptop\Google Drive\scripts\Pointcloud-processing\cloud.csv', index=False)
#rgb_broccoli.to_csv(r'C:\Users\laptop\Google Drive\scripts\Pointcloud-processing\rgb_cloud.csv', index=False)

cloud = PyntCloud.from_file(r'C:\Users\laptop\Google Drive\scripts\Pointcloud-processing\cloud.csv')
cloud.add_scalar_field("hsv")
voxelgrid_id = cloud.add_structure("voxelgrid", n_x=200, n_y=200, n_z=100)
new_cloud = cloud.get_sample("voxelgrid_nearest", voxelgrid_id=voxelgrid_id, as_PyntCloud=True)
new_cloud.to_file(r'C:\Users\laptop\Google Drive\scripts\Pointcloud-processing\out_file.csv',index=None)



dt = startin.DT()

dt.insert(only_broccoli)

b = dt.write_obj(r'C:\Users\laptop\Google Drive\scripts\Pointcloud-processing\broccoli_tria.obj')

# =============================================================================
# import geopandas as gpd
# 
# gdf = gpd.GeoDataFrame(only_broccoli, geometry=gpd.points_from_xy(x=only_broccoli.X, y=only_broccoli.Y))
# gdf.to_csv(r'C:\Users\laptop\Google Drive\scripts\Pointcloud-processing\b_cloud.csv',index=False)
# =============================================================================


#%%     clustering the points !!!
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
# https://en.wikipedia.org/wiki/DBSCAN

data = pd.read_csv(r'C:\Users\laptop\Google Drive\scripts\Pointcloud-processing\out_file.csv')
data = np.asarray(data)
x = np.array(data[:,0])
y = np.array(data[:,1])
z = np.array(data[:,2])

xn = (x - x.min()) / (x.max() - x.min())
yn = (y - y.min()) / (y.max() - y.min())
zn = (z - z.min()) / (z.max() - z.min())
xyz_nn = np.vstack([xn,yn,zn]).T

nbrs = NearestNeighbors(n_neighbors = 4, algorithm = 'auto').fit(data) #['auto', 'ball_tree', 'kd_tree', 'brute']
distances, indices = nbrs.kneighbors(data) #the indices of the nearest neighbors 

import sklearn
n=10
scores = np.zeros(n)
epslist = np.linspace(0.03,0.04,n)
for i in range(n):
    print(i)
    epsi = epslist[i]
    
    clustering = DBSCAN(eps=epsi, min_samples=4).fit(xyz_nn)
    labels = clustering.labels_
    
    scores[i] = sklearn.metrics.silhouette_score(xyz_nn, labels, metric='euclidean')
    
#clustering = DBSCAN(eps=epsi, min_samples=4).fit(data)
#labels = clustering.labels_

#nbrs = NearestNeighbors(n_neighbors = 4, algorithm = 'auto').fit(xyz_nn) #['auto', 'ball_tree', 'kd_tree', 'brute']
#distances, indices = nbrs.kneighbors(xyz_nn) #the indices of the nearest neighbors 

clustering = DBSCAN(eps=sadlfkjads, min_samples=4).fit(xyz_nn)
labels = clustering.labels_


#Rlist = np.random.randint(100,size=np.max(labels)+2)/100
#Glist = np.random.randint(100,size=np.max(labels)+2)/100
#Blist = np.random.randint(100,size=np.max(labels)+2)/100
#
#rgb = np.zeros(data.shape)
#
#for i in range(len(data)):
#    ind = labels[i]+1
#    rgb[i,0]=Rlist[labels[ind]]
#    rgb[i,1]=Glist[labels[ind]]
#    rgb[i,2]=Blist[labels[ind]]
    




