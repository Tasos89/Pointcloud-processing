# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 11:19:46 2020

@author: Tasos
"""
#################https://towardsdatascience.com/point-cloud-data-simple-approach-f3855fdc08f5    #############

#%% 
# import packages 
import os
import pptk
import numpy as np
from laspy.file import File
import math
import numba
from numba import jit
from skimage.color import rgb2lab
import cv2
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer

#%%

input_folder = r'C:\Users\laptop\Google Drive\Shared folder Tasos-VanBoven\Sample_data\Broccoli\35m' #35m Broccoli
# =============================================================================
# input_folder = r"C:\Users\laptop\Google Drive\Shared folder Tasos-VanBoven\Sample_data\Broccoli\AZ74_10m-0-1 - Cloud.las" #10m Broccoli
# data_las = File(folder_10m, mode = 'r')
# =============================================================================

las_files = []
for root, dirs, files in os.walk(input_folder, topdown=True):
    for name in files:
        if name[-4:] == ".las":
            las_files.append(os.path.join(root,name).replace("\\","/"))
data_las = File(las_files[45], mode='r') 

xyz = np.vstack([data_las.x, data_las.y, data_las.z]).transpose()
rgb = ((np.c_[data_las.Red, data_las.Green, data_las.Blue]) / 255.) / 255. #normalized
R = rgb[:,0]
G = rgb[:,1]
B = rgb[:,2]
x = xyz[:,0]
y = xyz[:,1]
z = xyz[:,2]

# We normalize the data because...https://stats.stackexchange.com/questions/287425/why-do-you-need-to-scale-data-in-knn
xn = (x - x.min()) / (x.max() - x.min())
yn = (y - y.min()) / (y.max() - y.min())
zn = (z - z.min()) / (z.max() - z.min())
xyz_nn = np.vstack([xn,yn,zn]).T

# Remove the median and divide by std = standarlize the data
median = np.median(a=xyz,axis=0)
std = np.std(a=xyz,axis=0)
x_std = (x - median[0])/std[0]
y_std = (y - median[1])/std[1]
z_std = (z - median[2])/std[2]
xyz_std = np.vstack([x_std,y_std,z_std]).T

# RBG values are already normalized between 0 and 1 

# from BGR to Lab # https://gist.github.com/bikz05/6fd21c812ef6ebac66e1
# The results using Lab instead of RGB was not so good, hence changing the color space didn't work for early crop data
# =============================================================================
# def func(t):
#     if (t > 0.008856):
#         return np.power(t, 1/3.0);
#     else:
#         return 7.787 * t + 16 / 116.0;
# 
# #Conversion Matrix
# matrix = [[0.412453, 0.357580, 0.180423],
#           [0.212671, 0.715160, 0.072169],
#           [0.019334, 0.119193, 0.950227]]
# 
# # RGB values lie between 0 to 1.0
# Lab_OpenCv = []
# Lab = np.zeros((len(rgb),3))
# for row in rgb:
#     cie = np.dot(matrix, row);
#     
#     cie[0] = cie[0] /0.950456;
#     cie[2] = cie[2] /1.088754; 
#     
#     # Calculate the L
#     L = 116 * np.power(cie[1], 1/3.0) - 16.0 if cie[1] > 0.008856 else 903.3 * cie[1];
#     
#     # Calculate the a 
#     a = 500*(func(cie[0]) - func(cie[1]));
#     
#     # Calculate the b
#     b = 200*(func(cie[1]) - func(cie[2]));
#     
#     #  Values lie between -128 < b <= 127, -128 < a <= 127, 0 <= L <= 100 
#     Lab = [b , a, L]; 
#     
#     # OpenCV Format
#     L = L * 255 / 100;
#     a = a + 128;
#     b = b + 128;
#     Lab_OpenCv.append([b,a,L])
# Lab_OpenCv = np.asarray(Lab_OpenCv)
# scaler = MinMaxScaler()
# Lab = scaler.fit_transform(Lab_OpenCv)
# Lab[:,2] = 0.0
# b = Lab[:,0]
# a = Lab[:,1]
# L = Lab[:,2]
# =============================================================================

#%%
# Define the ground points / ground filtering
import startin


#create a grid
xmin = np.floor(np.min(x))
ymin = np.floor(np.min(y))
xmax = np.ceil(np.max(x))
ymax = np.ceil(np.max(y))
x_grid = np.arange(xmin,xmax,1) #270 the step for capital X
y_grid = np.arange(ymin,ymax,1)
ground_points = []

for i in np.arange(0,len(x_grid)-1):
    for j in np.arange(0,len(y_grid)-1):  
        points_cell = np.where((x > x_grid[i]) & (y>y_grid[j]) & (x<x_grid[i+1]) & (y<y_grid[j+1])) #indices of points "inside" the cell
        z_points = z[points_cell[0]] #elevation of those points
        point_lowest = points_cell[0][np.argmin(z_points)]
        ground_points.append(point_lowest) #indices of the lowest points in every cell
        
x_lowest = x[ground_points]
y_lowest = y[ground_points]
z_lowest = z[ground_points]
x = np.delete(x,ground_points)
y = np.delete(y,ground_points)
z = np.delete(z,ground_points)
xyz_lowest = np.vstack((x_lowest,y_lowest,z_lowest)).T

dt = startin.DT()
dt.insert(xyz_lowest)

remaining = np.concatenate((x.reshape(-1,1),y.reshape(-1,1),z.reshape(-1,1)),axis=1)
points_outside = []
k = 0
check = 0
#for i in np.arange(21,22):
#and check = 0
while remaining.shape[0]>k:
    print(remaining.shape[0]-k)
    if check==1 and k==remaining.shape[0]:
        k=0
        check = 0
    
    Triangle = dt.locate(remaining[k,0],remaining[k,1])
    x_point0 = remaining[k,0]
    y_point0 = remaining[k,1]
    z_point0 = remaining[k,2]
    
    if len(Triangle)>0:
        tri_1 = dt.get_point(Triangle[0])
        tri_2 = dt.get_point(Triangle[1])
        tri_3 = dt.get_point(Triangle[2])
        points = np.array([tri_1,tri_2,tri_3])
        x_tri = points[:,0]
        y_tri = points[:,1]
        z_tri = points[:,2]
        Xm = np.mean(x_tri)
        Ym = np.mean(y_tri)
        Zm = np.mean(z_tri)
        x_point = x_point0-Xm
        y_point = y_point0-Ym
        z_point = z_point0-Zm
        xyz_point = np.array([x_point,y_point,z_point]).reshape(-1,1).T
        x_tri = (x_tri-Xm).reshape(-1,1)
        y_tri = (y_tri-Ym).reshape(-1,1)
        z_tri = (z_tri-Zm).reshape(-1,1)
        M = np.concatenate((x_tri,y_tri,z_tri),axis=1)
        MtM = np.dot(M.T,M)
        e, v = np.linalg.eig(MtM)
        emax = np.argmax(e)
        emin = np.argmin(e)
        emid = 3-emax-emin
        v2 = np.zeros([3,3])
        v2[:,0] = v[:,emax]
        v2[:,1] = v[:,emid]
        v2[:,2] = v[:,emin]
        tri_rot = np.dot(M,v2)
        point_rot = np.dot(xyz_point,v2)
        distance = np.abs(point_rot[0,2])
        beta = np.zeros(3)
        for j in np.arange(0,3):
            vertex_rot = tri_rot[j,:]
            hor_dist = np.sqrt(np.square(vertex_rot[0]-point_rot[0,0])+np.square(vertex_rot[1]-point_rot[0,1]))
            beta[j] = np.rad2deg(np.arctan(distance/hor_dist))
        alpha = np.max(beta)
        if distance<0.5 and alpha<20: #distance<15 and alpha<20
            dt.insert_one_pt(x_point0,y_point0,z_point0)
            remaining = np.delete(remaining,k,0)
            #k = 0
            k = k+1
            check = 1
            
        else:
            k = k+1
    else:
        points_outside.append(k)
        k = k+1
        
ground_points = dt.all_vertices()
ground_points = np.asarray(ground_points)
xyz_ground = ground_points[1:,:]

xyz_no_ground = remaining
rgb_no_gro

DTM = dt.write_obj(r"C:\Users\laptop\Google Drive\scripts\Pointcloud-processing\DTM.obj")

        
#%% 
# Nearest neighbors with normalized data. The confusion matrix return better results but the visualization was not so good

from sklearn.neighbors import NearestNeighbors #import NearestNeigbhors package
nbrs = NearestNeighbors(n_neighbors = 35, algorithm = 'kd_tree').fit(xyz_std) #['auto', 'ball_tree', 'kd_tree', 'brute']
distances, indices = nbrs.kneighbors(xyz_std) #the indices of the nearest neighbors 

# Nearest neigbors with standarlized data

# =============================================================================
# from sklearn.neighbors import NearestNeighbors #import NearestNeigbhors package
# nbrs = NearestNeighbors(n_neighbors = 35, algorithm = 'kd_tree').fit(xyz_std) #['auto', 'ball_tree', 'kd_tree', 'brute']
# distances, indices = nbrs.kneighbors(xyz_std) #the indices of the nearest neighbors 
# =============================================================================

#%% 
# extraction of geometrical features among the nearest neighbors 
# https://towardsdatascience.com/an-approach-to-choosing-the-number-of-components-in-a-principal-component-analysis-pca-3b9f3d6e73fe

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

for i in range(len(indices)):
    ind = indices[i]
    #coords = xyz[(ind),:] # for normalize data
    coords = xyz_std[(ind),:] #for standarlize
    


    x = coords[:,0]
    y = coords[:,1]
    z = coords[:,2]
# =============================================================================
#     R = coords[:,3]
#     G = coords[:,4]
#     B = coords[:,5]
# =============================================================================

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
     
# normalization of the geometrical features
# https://stats.stackexchange.com/questions/69157/why-do-we-need-to-normalize-data-before-principal-component-analysis-pca
omnivariance = np.asarray(omnivariance)
omn_n = (omnivariance -omnivariance.min()) / (omnivariance.max() - omnivariance.min())
l = np.asarray(linearity)
lin_n = (l -l.min()) / (l.max() - l.min())
p = np.asarray(planarity)
plan_n = (p -p.min()) / (p.max() - p.min())
s = np.asarray(scatter)
scat_n = (s -s.min()) / (s.max() - s.min())
an = np.asarray(anisotropy)
an_n = (an -an.min()) / (an.max() - an.min())
eig = np.asarray(eigenentropy)
eig_n = (eig -eig.min()) / (eig.max() - eig.min())
ch = np.asarray(change_curvature)
ch_cur_n = (ch -ch.min()) / (ch.max() - ch.min())
m_e = np.asarray(mean_elev)
mean_el_n = (m_e -m_e.min()) / (m_e.max() - m_e.min())
d_e = np.asarray(dif_elev)
dif_elev_n = (d_e -d_e.min()) / (d_e.max() - d_e.min())

#%% lecture 7 decession trees and pca (geoprocessing analysis)
# pca for the geometrical features to define the most important features with minmaxscaler features
# https://www.visiondummy.com/2014/05/feature-extraction-using-pca/
# https://chrisalbon.com/machine_learning/feature_engineering/feature_extraction_with_pca/ 
# https://towardsdatascience.com/an-approach-to-choosing-the-number-of-components-in-a-principal-component-analysis-pca-3b9f3d6e73fe

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

features = np.vstack((omn_n, lin_n, plan_n, scat_n, an_n, eig_n, ch_cur_n, mean_el_n, dif_elev_n, R, G, B)).T #normalize
features = np.vstack((omn_n, dif_elev_n, R, G, B)).T # till now the best combination of features!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
features = np.vstack((omn_n, dif_elev_n, L, a, b)).T #Lab

# we already have normalized data hence perhaps the minmaxscaler is redundant 
pca1 = PCA().fit(features)
plt.figure()
plt.plot(np.cumsum(pca1.explained_variance_ratio_))
plt.xlabel('Number of Features')
plt.ylabel('Variance (%)')
plt.title('Subjective Choose of Features')
plt.show()

# hence we can keep 5 attributes/features that have the variance of almost all the data ~100%
pca2 = PCA(n_components=5)
features_new = pca2.fit_transform(features)


#%%
#pca2with standarlized data
# https://chrisalbon.com/machine_learning/feature_engineering/feature_extraction_with_pca/
# https://medium.com/apprentice-journal/pca-application-in-machine-learning-4827c07a61db
# https://towardsdatascience.com/principal-component-analysis-for-dimensionality-reduction-115a3d157bad

from sklearn import decomposition, datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

features = np.vstack((omn_n, lin_n, plan_n, scat_n, an_n, eig_n, ch_cur_n, mean_el_n, dif_elev_n, R, G, B)).T
features = np.asarray(features)
sc = StandardScaler()
#fit the scaler to the features and transform
features = sc.fit_transform(features)
#create a pca object with the 8 components as parameter
pca = decomposition.PCA(n_components=5)
#fit the PCA and transform the data
features_pca = pca.fit_transform(features_std)
features_pca.shape


plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Features')
plt.ylabel('Variance (%)')
plt.title('Subjective Choose of Features')
plt.show()

