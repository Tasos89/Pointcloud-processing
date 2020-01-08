#%% Broccoli
import os
import pptk
import numpy as np
from laspy.file import File
import math

Broccoli = r"C:\Users\laptop\Google Drive\Shared folder Tasos-VanBoven\Sample_data\Broccoli\classes\broccoli.las"
           
data_las = File(Broccoli, mode='r') 

xyz_B = np.vstack([data_las.X, data_las.Y, data_las.Z]).transpose()
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

from sklearn.neighbors import NearestNeighbors #import NearestNeigbhors package
nbrs = NearestNeighbors(n_neighbors = 30, algorithm = 'kd_tree').fit(xyz_B) #['auto', 'ball_tree', 'kd_tree', 'brute']
distances, indices_B = nbrs.kneighbors(xyz_B) #the indices of the nearest neighbors 

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
    coords = xyz_B[(ind),:]
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
    ei = -(e[0]*math.log(e[0])+e[1]*math.log(e[1])+e[2]*math.log(e[2]))
    eigenentropy.append(ei)
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
eig = np.asarray(eigenentropy)
ch = np.asarray(change_curvature)
m_e = np.asarray(mean_elev)
d_e = np.asarray(dif_elev)

#standardization is used for PCA. preprocessing.StandardScaler does not work for me, hence manually..
omn_std_B = (omnivariance-omnivariance.mean())/omnivariance.std()
lin_std_B = (l-l.mean())/l.std()
plan_std_B = (p-p.mean())/p.std()
scat_std_B = (s-s.mean())/s.std()
an_std_B = (an-an.mean())/an.std()
eig_std_B = (eig-eig.mean())/eig.std()
ch_cur_std_B = (ch-ch.mean())/ch.std()
mean_el_std_B = (m_e-m_e.mean())/m_e.std()
dif_elev_std_B = (d_e-d_e.mean())/d_e.std()

Broccoli_LABEL = 1.*np.ones(len(xyz_B))
features_B = np.vstack((omn_std_B, lin_std_B, plan_std_B, scat_std_B, an_std_B, eig_std_B, ch_cur_std_B, mean_el_std_B, dif_elev_std_B, r_std_B, g_std_B, b_std_B,Broccoli_LABEL)).T

#%% Grass

Grass = r"C:\Users\laptop\Google Drive\Shared folder Tasos-VanBoven\Sample_data\Broccoli\classes\grass.las"
           
data_las = File(Grass, mode='r') 

xyz_G = np.vstack([data_las.X, data_las.Y, data_las.Z]).transpose()
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

from sklearn.neighbors import NearestNeighbors #import NearestNeigbhors package
nbrs = NearestNeighbors(n_neighbors = 33, algorithm = 'kd_tree').fit(xyz_G) #['auto', 'ball_tree', 'kd_tree', 'brute']
distances, indices_G = nbrs.kneighbors(xyz_G) #the indices of the nearest neighbors 

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

for i in range(len(indices_G)):
    ind = indices_G[i]
    coords = xyz_G[(ind),:]
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
    ei = -(e[0]*math.log(e[0])+e[1]*math.log(e[1])+e[2]*math.log(e[2]))
    eigenentropy.append(ei)
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
eig = np.asarray(eigenentropy)
ch = np.asarray(change_curvature)
m_e = np.asarray(mean_elev)
d_e = np.asarray(dif_elev)

#standardization is used for PCA. preprocessing.StandardScaler does not work for me, hence manually..
omn_std_G = (omnivariance-omnivariance.mean())/omnivariance.std()
lin_std_G = (l-l.mean())/l.std()
plan_std_G = (p-p.mean())/p.std()
scat_std_G = (s-s.mean())/s.std()
an_std_G = (an-an.mean())/an.std()
eig_std_G = (eig-eig.mean())/eig.std()
ch_cur_std_G = (ch-ch.mean())/ch.std()
mean_el_std_G = (m_e-m_e.mean())/m_e.std()
dif_elev_std_G = (d_e-d_e.mean())/d_e.std()

Grass_LABEL = 2.*np.ones(len(xyz_G))
features_G = np.vstack((omn_std_G, lin_std_G, plan_std_G, scat_std_G, an_std_G, eig_std_G, ch_cur_std_G, mean_el_std_G, dif_elev_std_G, r_std_G, g_std_G, b_std_G,Grass_LABEL)).T

#%% Soil


Soil = r"C:\Users\laptop\Google Drive\Shared folder Tasos-VanBoven\Sample_data\Broccoli\classes\soil.las"
           
data_las = File(Soil, mode='r') 

xyz_S = np.vstack([data_las.X, data_las.Y, data_las.Z]).transpose()
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

from sklearn.neighbors import NearestNeighbors #import NearestNeigbhors package
nbrs = NearestNeighbors(n_neighbors = 33, algorithm = 'kd_tree').fit(xyz_S) #['auto', 'ball_tree', 'kd_tree', 'brute']
distances, indices_S = nbrs.kneighbors(xyz_S) #the indices of the nearest neighbors 

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

for i in range(len(indices_S)):
    ind = indices_S[i]
    coords = xyz_S[(ind),:]
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
    ei = -(e[0]*math.log(e[0])+e[1]*math.log(e[1])+e[2]*math.log(e[2]))
    eigenentropy.append(ei)
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
eig = np.asarray(eigenentropy)
ch = np.asarray(change_curvature)
m_e = np.asarray(mean_elev)
d_e = np.asarray(dif_elev)

#standardization is used for PCA. preprocessing.StandardScaler does not work for me, hence manually..
omn_std_S = (omnivariance-omnivariance.mean())/omnivariance.std()
lin_std_S = (l-l.mean())/l.std()
plan_std_S = (p-p.mean())/p.std()
scat_std_S = (s-s.mean())/s.std()
an_std_S = (an-an.mean())/an.std()
eig_std_S = (eig-eig.mean())/eig.std()
ch_cur_std_S = (ch-ch.mean())/ch.std()
mean_el_std_S = (m_e-m_e.mean())/m_e.std()
dif_elev_std_S = (d_e-d_e.mean())/d_e.std()

Soil_LABEL = 3.*np.ones(len(xyz_S))
features_S = np.vstack((omn_std_S, lin_std_S, plan_std_S, scat_std_S, an_std_S, eig_std_S, ch_cur_std_S, mean_el_std_S, dif_elev_std_S, r_std_S, g_std_S, b_std_S,Soil_LABEL)).T

#%%
feat_classes = np.vstack((features_B,features_G,features_S))
#labelsClasses = [(ones(length(linearity2),1)); 2*(ones(length(linearity3),1)); 3*(ones(length(linearity4),1)); 4*(ones(length(linearity5),1))];
label_classes = np.concatenate((np.ones(len(indices_B)),2*np.ones(len(indices_G)),3*np.ones(len(indices_S))))
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

#Mdl = BaggingClassifier(30,feat_classes,label_classes)

clf = RandomForestClassifier(n_estimators=100, max_features=3,bootstrap=True)
scores_fit = clf.fit(feat_classes,label_classes)
scores = pd.DataFrame(clf.predict_proba(feat_classes))
v = pptk.viewer(label_classes,scores)




# =============================================================================
# from sklearn.linear_model import LogisticRegression
# model =  LogisticRegression()
# predict = model.predict(x)
# 
# =============================================================================






