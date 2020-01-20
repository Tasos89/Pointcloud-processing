#Iris example https://towardsdatascience.com/decision-tree-in-python-b433ae57fb93
#%% broccoli label=1
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO 
from IPython.display import Image 
from pydot import graph_from_dot_data
import pandas as pd
import numpy as np
import pptk
import math
from laspy.file import File

Broccoli = r"C:\Users\laptop\Google Drive\Shared folder Tasos-VanBoven\Sample_data\Broccoli\classes\broccoli3.las"
           
data_las = File(Broccoli, mode='r') 

xyz_B = np.vstack([data_las.x, data_las.y, data_las.z]).transpose()
rgb_B = ((np.c_[data_las.Red, data_las.Green, data_las.Blue]) / 255.) / 255. #normalized
R = rgb_B[:,0]
G = rgb_B[:,1]
B = rgb_B[:,2]
median_rgb = np.median(a=rgb_B,axis=0)
std_rgb = np.std(a=rgb_B,axis=0)
r_std_B = (R - median_rgb[0])/std_rgb[0]
g_std_B = (G - median_rgb[1])/std_rgb[1]
b_std_B = (B - median_rgb[2])/std_rgb[2]
rgb_std_B = np.vstack([r_std_B,g_std_B,b_std_B]).T

#normalize the rgb 
r_n_B = (R - R.min()) / (R.max() - R.min())
g_n_B = (G - G.min()) / (G.max() - G.min())
b_n_B = (B - B.min()) / (B.max() - B.min())

#normalize the xyz
transformer = Normalizer().fit(xyz_B)
xyz_BB = transformer.transform(xyz_B)

#standarlize the xyz
median_xyz = np.median(a=xyz_B,axis=0)
std_xyz = np.std(a=xyz_B,axis=0)
x_std_B = (xyz_B[:,0] - median_xyz[0])/std_xyz[0]
y_std_B = (xyz_B[:,1] - median_xyz[1])/std_xyz[1]
z_std_B = (xyz_B[:,2] - median_xyz[2])/std_xyz[2]
xyz_std_B = np.vstack([x_std_B,y_std_B,z_std_B]).T

xn = (data_las.x - data_las.x.min()) / (data_las.x.max() - data_las.x.min())
yn = (data_las.y - data_las.y.min()) / (data_las.y.max() - data_las.y.min())
zn = (data_las.z - data_las.z.min()) / (data_las.z.max() - data_las.z.min())
xyz_nn = np.vstack([xn,yn,zn]).T


# =============================================================================
# from sklearn.neighbors import NearestNeighbors #import NearestNeigbhors package
# nbrs = NearestNeighbors(n_neighbors = 35, algorithm = 'kd_tree').fit(xyz_std_BB) #['auto', 'ball_tree', 'kd_tree', 'brute']
# distances, indices_B = nbrs.kneighbors(xyz_std_B) #the indices of the nearest neighbors 
# =============================================================================

from sklearn.neighbors import NearestNeighbors #import NearestNeigbhors package
nbrs = NearestNeighbors(n_neighbors = 35, algorithm = 'kd_tree').fit(xyz_nn) #['auto', 'ball_tree', 'kd_tree', 'brute']
distances, indices_B = nbrs.kneighbors(xyz_nn) #the indices of the nearest neighbors 

linearity = []
planarity = []
scatter = []
omnivariance = []
anisotropy = []
eigenentropy = []
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


#normalization of the data 
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
# 
# Broccoli_LABEL = 1.*np.ones(len(xyz_B))
# features_B = np.vstack((omn_n_B, dif_elev_n_B, L_B, a_B, b_B, Broccoli_LABEL)).T # Lab !!!!!!!!!!
# =============================================================================


Broccoli_LABEL = 1.*np.ones(len(xyz_B))
features_B = np.vstack((dif_elev_n_B, omn_n_B, scat_n_B, an_n_B, dif_elev_n_B, r_n_B, g_n_B, b_n_B, Broccoli_LABEL)).T # When use rgb std works better


features_B = np.vstack((omn_n_B, dif_elev_n_B, R, G ,B, Broccoli_LABEL)).T
#features_B = np.vstack((omn_n_B, lin_n_B, plan_n_B, scat_n_B, an_n_B, ch_cur_n_B, mean_el_n_B, dif_elev_n_B, r_n_B, g_n_B, b_n_B)).T
#features_B = np.vstack((omn_n_B, lin_n_B, plan_n_B, scat_n_B, an_n_B, ch_cur_n_B, mean_el_n_B, dif_elev_n_B, L_B, a_B, b_B)).T

#pca
# =============================================================================
# features_B = pca2.fit_transform(features_B)
# features_B = np.insert(features_B, 5, Broccoli_LABEL, axis=1)
# =============================================================================





# =============================================================================
# Broccoli_LABEL = 1.*np.ones(len(xyz_B))
# #features_B = np.vstack((, lin_std_B, plan_std_B, scat_std_B, an_std_B, ch_cur_std_B, mean_el_std_B, dif_elev_std_B, r_std_B, g_std_B, b_std_B,Broccoli_LABEL)).T
# #features_B = np.vstack((omn_std_B, lin_std_B, plan_std_B, scat_std_B, an_std_B, ch_cur_std_B, dif_elev_std_B, r_std_B, g_std_B, b_std_B,Broccoli_LABEL)).T
# #features_B = np.vstack((omn_std_B, lin_std_B, plan_std_B, scat_std_B, an_std_B, ch_cur_std_B, dif_elev_std_B,Broccoli_LABEL)).T
# #features_B = np.vstack((omn_std_B, lin_std_B, plan_std_B, ch_cur_std_B, r_std_B, g_std_B, b_std_B, Broccoli_LABEL)).T
# #features_B = np.vstack((dif_elev_std_B, omn_std_B, scat_std_B, an_std_B, r_std_B, g_std_B, b_std_B, Broccoli_LABEL)).T
# features_B = np.vstack((dif_elev_std_B, omn_std_B, scat_std_B, an_std_B, Broccoli_LABEL)).T
# =============================================================================



#%% Grass label=2

Grass = r"C:\Users\laptop\Google Drive\Shared folder Tasos-VanBoven\Sample_data\Broccoli\classes\grass3.las"
           
data_las = File(Grass, mode='r') 

xyz_G = np.vstack([data_las.x, data_las.y, data_las.z]).transpose()
rgb_G = ((np.c_[data_las.Red, data_las.Green, data_las.Blue]) / 255.) / 255. #normalized
R = rgb_G[:,0]
G = rgb_G[:,1]
B = rgb_G[:,2]
median_rgb = np.median(a=rgb_G,axis=0)
std_rgb = np.std(a=rgb_G,axis=0)
r_std_G = (R - median_rgb[0])/std_rgb[0]
g_std_G = (G - median_rgb[1])/std_rgb[1]
b_std_G = (B - median_rgb[2])/std_rgb[2]
rgb_std_G = np.vstack([r_std_G,g_std_G,b_std_G]).T

#normalize the rgb
r_n_G = (R - R.min()) / (R.max() - R.min())
g_n_G = (G - G.min()) / (G.max() - G.min())
b_n_G = (B - B.min()) / (B.max() - B.min())

#normalize the xyz
transformer = Normalizer().fit(xyz_G)
xyz_GG = transformer.transform(xyz_G)

#standarlize the xyz
median_xyz = np.median(a=xyz_G,axis=0)
std_xyz = np.std(a=xyz_G,axis=0)
x_std_G = (xyz_G[:,0] - median_xyz[0])/std_xyz[0]
y_std_G = (xyz_G[:,1] - median_xyz[1])/std_xyz[1]
z_std_G = (xyz_G[:,2] - median_xyz[2])/std_xyz[2]
xyz_std_G = np.vstack([x_std_G,y_std_G,z_std_G]).T

xn = (data_las.x - data_las.x.min()) / (data_las.x.max() - data_las.x.min())
yn = (data_las.y - data_las.y.min()) / (data_las.y.max() - data_las.y.min())
zn = (data_las.z - data_las.z.min()) / (data_las.z.max() - data_las.z.min())
xyz_nn = np.vstack([xn,yn,zn]).T

# =============================================================================
# from sklearn.neighbors import NearestNeighbors #import NearestNeigbhors package
# nbrs = NearestNeighbors(n_neighbors = 35, algorithm = 'kd_tree').fit(xyz_std_G) #['auto', 'ball_tree', 'kd_tree', 'brute']
# distances, indices_G = nbrs.kneighbors(xyz_std_G) #the indices of the nearest neighbors 
# =============================================================================

from sklearn.neighbors import NearestNeighbors #import NearestNeigbhors package
nbrs = NearestNeighbors(n_neighbors = 35, algorithm = 'kd_tree').fit(xyz_nn) #['auto', 'ball_tree', 'kd_tree', 'brute']
distances, indices_G = nbrs.kneighbors(xyz_nn) #the indices of the nearest neighbors 

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

#normalization of the data 
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
# 
# Grass_LABEL = 2.*np.ones(len(xyz_G))
# features_G = np.vstack((omn_n_G, dif_elev_n_G, L_G, a_G, b_G, Grass_LABEL)).T # Lab !!!!!!!!!!
# =============================================================================

Grass_LABEL = 2.*np.ones(len(xyz_G))
features_G = np.vstack(( dif_elev_n_G, omn_n_G, scat_n_G, an_n_G,  dif_elev_n_G, r_n_G, g_n_G, b_n_G, Grass_LABEL)).T


features_G = np.vstack((omn_n_G, dif_elev_n_G, R, G, B, Grass_LABEL)).T
#features_G = np.vstack((omn_n_G, lin_n_G, plan_n_G, scat_n_G, an_n_G, ch_cur_n_G, mean_el_n_G, dif_elev_n_G, r_n_G, g_n_G, b_n_G)).T

#pca
# =============================================================================
# features_G = pca2.fit_transform(features_G)
# features_G = np.insert(features_G, 5, Grass_LABEL, axis=1)
# =============================================================================



# =============================================================================
# Grass_LABEL = 2.*np.ones(len(xyz_G))
# #features_G = np.vstack((omn_std_G, lin_std_G, plan_std_G, scat_std_G, an_std_G, ch_cur_std_G, mean_el_std_G, dif_elev_std_G, r_std_G, g_std_G, b_std_G,Grass_LABEL)).T
# #features_G = np.vstack((omn_std_G, lin_std_G, plan_std_G, scat_std_G, an_std_G, ch_cur_std_G, dif_elev_std_G, r_std_G, g_std_G, b_std_G,Grass_LABEL)).T
# #features_G = np.vstack((omn_std_G, lin_std_G, plan_std_G, scat_std_G, an_std_G, ch_cur_std_G, dif_elev_std_G,Grass_LABEL)).T
# #features_G = np.vstack((omn_std_G, lin_std_G, plan_std_G, ch_cur_std_G, r_std_G, g_std_G, b_std_G, Grass_LABEL)).T
# #features_G = np.vstack(( dif_elev_std_G, omn_std_G, scat_std_G, an_std_G, r_std_G, g_std_G, b_std_G, Grass_LABEL)).T
# features_G = np.vstack(( dif_elev_std_G, omn_std_G, scat_std_G, an_std_G, Grass_LABEL)).T
# =============================================================================



#%% Soil label=3


Soil = r"C:\Users\laptop\Google Drive\Shared folder Tasos-VanBoven\Sample_data\Broccoli\classes\soil4.las"
           
data_las = File(Soil, mode='r') 

xyz_S = np.vstack([data_las.x, data_las.y, data_las.z]).transpose()
rgb_S = ((np.c_[data_las.Red, data_las.Green, data_las.Blue]) / 255.) / 255. #normalized
R = rgb_S[:,0]
G = rgb_S[:,1]
B = rgb_S[:,2]
median_rgb = np.median(a=rgb_S,axis=0)
std_rgb = np.std(a=rgb_S,axis=0)
r_std_S = (R - median_rgb[0])/std_rgb[0]
g_std_S = (G - median_rgb[1])/std_rgb[1]
b_std_S = (B - median_rgb[2])/std_rgb[2]
rgb_std_S = np.vstack([r_std_S,g_std_S,b_std_S]).T

#normalize the rgb
r_n_S = (R - R.min()) / (R.max() - R.min())
g_n_S = (G - G.min()) / (G.max() - G.min())
b_n_S = (B - B.min()) / (B.max() - B.min())

#normalize the xyz
transformer = Normalizer().fit(xyz_S)
xyz_SS = transformer.transform(xyz_S)

#standarlize the xyz
median_xyz = np.median(a=xyz_B,axis=0)
std_xyz = np.std(a=xyz_B,axis=0)
x_std_S = (xyz_S[:,0] - median_xyz[0])/std_xyz[0]
y_std_S = (xyz_S[:,1] - median_xyz[1])/std_xyz[1]
z_std_S = (xyz_S[:,2] - median_xyz[2])/std_xyz[2]
xyz_std_S = np.vstack([x_std_S,y_std_S,z_std_S]).T

xn = (data_las.x - data_las.x.min()) / (data_las.x.max() - data_las.x.min())
yn = (data_las.y - data_las.y.min()) / (data_las.y.max() - data_las.y.min())
zn = (data_las.z - data_las.z.min()) / (data_las.z.max() - data_las.z.min())
xyz_nn = np.vstack([xn,yn,zn]).T


# =============================================================================
# from sklearn.neighbors import NearestNeighbors #import NearestNeigbhors package
# nbrs = NearestNeighbors(n_neighbors = 35, algorithm = 'kd_tree').fit(xyz_std_S) #['auto', 'ball_tree', 'kd_tree', 'brute']
# distances, indices_S = nbrs.kneighbors(xyz_std_S) #the indices of the nearest neighbors 
# =============================================================================

from sklearn.neighbors import NearestNeighbors #import NearestNeigbhors package
nbrs = NearestNeighbors(n_neighbors = 35, algorithm = 'kd_tree').fit(xyz_nn) #['auto', 'ball_tree', 'kd_tree', 'brute']
distances, indices_S = nbrs.kneighbors(xyz_nn) #the indices of the nearest neighbors 

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

#standardization is used for PCA. preprocessing.StandardScaler does not work for me, hence manually..
omn_std_S = (omnivariance-omnivariance.mean())/omnivariance.std()
lin_std_S = (l-l.mean())/l.std()
plan_std_S = (p-p.mean())/p.std()
scat_std_S = (s-s.mean())/s.std()
an_std_S = (an-an.mean())/an.std()
ch_cur_std_S = (ch-ch.mean())/ch.std()
mean_el_std_S = (m_e-m_e.mean())/m_e.std()
dif_elev_std_S = (d_e-d_e.mean())/d_e.std()

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
# 
# Soil_LABEL = 3.*np.ones(len(xyz_S))
# features_S = np.vstack((omn_n_S, dif_elev_n_S, L_S, a_S, b_S, Soil_LABEL)).T # Lab !!!!!!!!!!
# =============================================================================

Soil_LABEL = 3.*np.ones(len(xyz_S))
features_S = np.vstack((dif_elev_n_S, omn_n_S, scat_n_S, an_n_S,  dif_elev_n_S, r_n_S, g_n_S, b_n_S, Soil_LABEL)).T


features_S = np.vstack((omn_n_S, dif_elev_n_S, R, G, B, Soil_LABEL)).T
#features_S = np.vstack((omn_n_S, lin_n_S, plan_n_S, scat_n_S, an_n_S, ch_cur_n_S, mean_el_n_S, dif_elev_n_S, r_n_S, g_n_S, b_n_S)).T

#pca
# =============================================================================
# features_S = pca2.fit_transform(features_S)
# features_S = np.insert(features_S, 5, Soil_LABEL, axis=1)
# =============================================================================


# =============================================================================
# Soil_LABEL = 3.*np.ones(len(xyz_S))
# #features_S = np.vstack((omn_std_S, lin_std_S, plan_std_S, scat_std_S, an_std_S, ch_cur_std_S, mean_el_std_S, dif_elev_std_S, r_std_S, g_std_S, b_std_S,Soil_LABEL)).T
# #features_S = np.vstack((omn_std_S, lin_std_S, plan_std_S, scat_std_S, an_std_S, ch_cur_std_S, dif_elev_std_S, r_std_S, g_std_S, b_std_S,Soil_LABEL)).T
# #features_S = np.vstack((omn_std_S, lin_std_S, plan_std_S, scat_std_S, an_std_S, ch_cur_std_S, dif_elev_std_S,Soil_LABEL)).T
# #features_S = np.vstack((omn_std_S, lin_std_S, plan_std_S, ch_cur_std_S, r_std_S, g_std_S, b_std_S, Soil_LABEL)).T
# #features_S = np.vstack((dif_elev_std_S, omn_std_S, scat_std_S, an_std_S, r_std_S, g_std_S, b_std_S, Soil_LABEL)).T
# features_S = np.vstack((dif_elev_std_S, omn_std_S, scat_std_S, an_std_S, Soil_LABEL)).T
# =============================================================================



#%% Decision tree
# https://towardsdatascience.com/decision-tree-in-python-b433ae57fb93
# https://www.datacamp.com/community/tutorials/random-forests-classifier-python
# https://towardsdatascience.com/discretisation-using-decision-trees-21910483fa4b


#feature_names = ['omnivariance','linearity','planarity','scattering','anisotropy','chan_curvature','mean_elev','dif_elev','R','G','B','label']
#feature_names = ['omnivariance','linearity','planarity','scattering','anisotropy','chan_curvature','dif_elev','R','G','B','label']
#feature_names = ['omnivariance','linearity','planarity','scattering','anisotropy','chan_curvature','dif_elev','label']
#feature_names = ['omnivariance','linearity','planarity','chan_curvature','R','G','B','label']
#feature_names = ['dif_elev','omnivariance','scattering','anisotropy','dif_elev', 'R', 'G', 'B', 'label']
feature_names = ['omn','dif_elev','R','G','B','label']
#feature_names = ['omnivariance','difference in elevation','L','a','b','label'] #Lab




labels = ['Broccoli','Grass','Soil']

X = np.vstack((features_B,features_G,features_S))
X = pd.DataFrame(X, columns=feature_names)
y = pd.DataFrame(0, index=np.arange(len(X)),columns=labels)
y['Broccoli'][X['label']==1]=1
y['Grass'][X['label']==2]=1
y['Soil'][X['label']==3]=1

X = X.drop('label',axis=1) #The new dataframe reduced the quality of the prediction (see confision matrix)


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
confusion_matrix(species, predictions) # 2123/218  89% precise predictions

#%% 1 test the clasification with all the geometrical features

last_column = np.empty(len(indices),dtype=object)
np.asarray(last_column)

#features = np.vstack((omn_std, lin_std, plan_std, scat_std, an_std, ch_cur_std, mean_el_std, dif_elev_std, r_std, g_std, b_std)).T
#features = np.vstack((omn_std, lin_std, plan_std, scat_std, an_std, ch_cur_std, dif_elev_std, r_std, g_std, b_std)).T
#features = np.vstack((omn_std, lin_std, plan_std,ch_cur_std, r_std, g_std, b_std)).T
#features = np.vstack(((dif_elev_std, omn_std, scat_std, an_std, r_std, g_std, b_std))).T

#features = np.vstack(((dif_elev_std, omn_std, scat_std,an_std, ))).T #standarlized features
#features = np.vstack((dif_elev_n, omn_n, scat_n,an_n,dif_elev_n, r_n, g_n, b_n) ).T #Normalized features




#feature_names = ['omnivariance','linearity','planarity','scattering','anisotropy','eigenotropy','chan_curvature','mean_elev','dif_elev','R','G','B']
#feature_names = ['omnivariance','linearity','planarity','scattering','anisotropy','chan_curvature', 'dif_elev','R','G','B']
#feature_names = ['omnivariance','linearity','planarity','chan_curvature','R','G','B']
#feature_names = ['dif_elev','omnivariance','scattering','anisotropy','dif-elev', 'R', 'G', 'B']
feature_names = ['omn','dif_elev','R','G','B'] #RGB
#feature_names = ['omnivariance','difference in elevation','L','a','b'] #Lab

xx = pd.DataFrame(features, columns=feature_names)

yy_pred = clr.predict(xx)
v = pptk.viewer(xyz,yy_pred)

#%% Random forest
# https://blog.goodaudience.com/machine-learning-using-decision-trees-and-random-forests-in-python-with-code-e50f6e14e19f
from sklearn.ensemble import RandomForestClassifier
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
#keep only the broccoli by keeping the red points
red_indices = np.where(yyyy_pred[:,0]==1)
red_indices = np.reshape(red_indices,-1,1)
only_broccoli = xyz[red_indices,:]
rgb_broccoli = rgb[red_indices,:]
v = pptk.viewer(only_broccoli,rgb_broccoli) 
v.set(point_size=0.01)

# voxelization 
# https://github.com/daavoo/pyntcloud 
# https://pyntcloud.readthedocs.io/en/latest/points.html
# https://medium.com/@shakasom/how-to-convert-latitude-longtitude-columns-in-csv-to-geometry-column-using-python-4219d2106dea
# https://github.com/mcoder2014/voxelization  !!!!!!!!!!!!!!!!!!
from pyntcloud import PyntCloud
import pandas as pd

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


#clustering the points !!!
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
import pptk
data = pd.read_csv(r'C:\Users\laptop\Google Drive\scripts\Pointcloud-processing\out_file.csv')
data = np.asarray(data)

clustering = DBSCAN(eps=0.4, min_samples=10).fit(data)
labels = clustering.labels_

v = pptk.viewer(data)
R = np.zeros(len(labels))
G = np.zeros(len(labels))
B = np.zeros(len(labels))

for i in range(len(labels)):
    R[i] = labels[i]*10
    B[i] = labels[i]*10
    G[i] = labels[i]*10
    
rgb = np.vstack((R,G,B)).T
