#################https://towardsdatascience.com/point-cloud-data-simple-approach-f3855fdc08f5    #############
#PCA model
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

input
input_folder = r'C:\Users\laptop\Google Drive\Shared folder Tasos-VanBoven\Sample_data\Broccoli\35m' #35m Broccoli
#input_folder = r"C:\Users\laptop\Google Drive\Shared folder Tasos-VanBoven\Sample_data\Broccoli\AZ74_10m-0-1 - Cloud.las" #10m Broccoli
#data_las = File(input_folder, mode = 'r')

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
# Nearest neighbors with normalized data. The confusion matrix return better results but the visualization was not so good

from sklearn.neighbors import NearestNeighbors #import NearestNeigbhors package
nbrs = NearestNeighbors(n_neighbors = 35, algorithm = 'kd_tree').fit(xyz_nn) #['auto', 'ball_tree', 'kd_tree', 'brute']
distances, indices = nbrs.kneighbors(xyz_nn) #the indices of the nearest neighbors 

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
#eigenentropy = []
change_curvature = []
dif_elev = []
mean_elev = []
omnivariance = []

for i in range(len(indices)):
    ind = indices[i]
    coords = xyz_nn[(ind),:] # for normalize data
    #coords = xyz_std[(ind),:] #for standarlize
    


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
    #ei = -(e[0]*math.log(e[0])+e[1]*math.log(e[1])+e[2]*math.log(e[2]))
    #eigenentropy.append(ei)
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
#eig = np.asarray(eigenentropy)
#eig_n = (eig -eig.min()) / (eig.max() - eig.min())
ch = np.asarray(change_curvature)
ch_cur_n = (ch -ch.min()) / (ch.max() - ch.min())
m_e = np.asarray(mean_elev)
mean_el_n = (m_e -m_e.min()) / (m_e.max() - m_e.min())
d_e = np.asarray(dif_elev)
dif_elev_n = (d_e -d_e.min()) / (d_e.max() - d_e.min())

#standardization is used for PCA? preprocessing.StandardScaler does not work for me, hence manually..
# =============================================================================
# omn_std = (omnivariance-omnivariance.mean())/omnivariance.std()
# lin_std = (l-l.mean())/l.std()
# plan_std = (p-p.mean())/p.std()
# scat_std = (s-s.mean())/s.std()
# an_std = (an-an.mean())/an.std()
# eig_std = (eig-eig.mean())/eig.std()
# ch_cur_std = (ch-ch.mean())/ch.std()
# mean_el_std = (m_e-m_e.mean())/m_e.std()
# dif_elev_std = (d_e-d_e.mean())/d_e.std()
# =============================================================================

#visualization
# =============================================================================
# v = pptk.viewer(xyz,lin_std)
# v.set(point_size=0.02)
# v.capture('Linearity.png')
# 
# v = pptk.viewer(xyz,plan_std)
# v.set(point_size=7)
# v.capture('Planarity.png')
# 
# v = pptk.viewer(xyz,scat_std)
# v.set(point_size=7)
# v.capture('Scattering.png')
# 
# v = pptk.viewer(xyz,an_std)
# v.set(point_size=7)
# v.capture('Anisotropy.png')
# 
# v = pptk.viewer(xyz,eig_std)
# v.set(point_size=10)
# v.capture('Eigenotropy.png')
# 
# 
# v = pptk.viewer(xyz,ch_cur_std)
# v.set(point_size=7)
# v.capture('Change_of_Curvature.png')
# 
# v = pptk.viewer(xyz,mean_el_std)
# v.set(point_size=7)
# v.capture('Mean_elevation.png')
# 
# v = pptk.viewer(xyz,dif_elev_std)
# v.set(point_size=7)
# v.capture('Elevation_Difference.png')
# 
# v = pptk.viewer(xyz,omn_std)
# v.set(point_size=7)
# v.capture('Omnivariance.png')
# =============================================================================

#%% lecture 7 decession trees and pca (geoprocessing analysis)
#pca for the geometrical features to define the most important features with minmaxscaler features
# https://www.visiondummy.com/2014/05/feature-extraction-using-pca/
# https://chrisalbon.com/machine_learning/feature_engineering/feature_extraction_with_pca/ 
# https://towardsdatascience.com/an-approach-to-choosing-the-number-of-components-in-a-principal-component-analysis-pca-3b9f3d6e73fe

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#features = np.vstack((omn_std, lin_std, plan_std, scat_std, an_std, eig_std, ch_cur_std, mean_el_std, dif_elev_std, r_std, g_std, b_std)).T
features = np.vstack((omn_n, lin_n, plan_n, scat_n, an_n, eig_n, ch_cur_n, mean_el_n, dif_elev_n, r_n, g_n, b_n)).T #normalize
#features = np.vstack((omn_n, lin_n, plan_n, scat_n, an_n, eig_n, ch_cur_n, mean_el_n, dif_elev_n, L, a, b)).T #normalize
features = np.vstack((omn_n, lin_n, plan_n, scat_n, an_n, eig_n, ch_cur_n, mean_el_n, dif_elev_n, R, G, B)).T
features = np.vstack((omn_n, dif_elev_n, R, G, B)).T # till now the best combination of features!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
features = np.vstack((omn_n, dif_elev_n, L, a, b)).T #Lab







#data = np.concatenate((features),axis=0)
# =============================================================================
# scaler = MinMaxScaler(feature_range=[0,1])
# data_rescaled = scaler.fit_transform(features)
# pca = PCA().fit(data_rescaled)
# =============================================================================

pca = PCA().fit(features)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Features')
plt.ylabel('Variance (%)')
plt.title('Subjective Choose of Features')
plt.show()
#plt.savefig('pca.png')

# we already have normalized data hence perhaps the minmaxscaler is redundant 
pca1 = PCA().fit(features)
plt.figure()
plt.plot(np.cumsum(pca2.explained_variance_ratio_))
plt.xlabel('Number of Features')
plt.ylabel('Variance (%)')
plt.title('After the redundancy')
plt.show()
plt.savefig('After_pca.png')

#hence we can keep 5 attributes/features that have the variance of almost all the data ~100%
pca2 = PCA(n_components=5)
features_new = pca2.fit_transform(features)


#%%
#pca2with standarlized data
# https://chrisalbon.com/machine_learning/feature_engineering/feature_extraction_with_pca/
# https://medium.com/apprentice-journal/pca-application-in-machine-learning-4827c07a61db
# https://towardsdatascience.com/principal-component-analysis-for-dimensionality-reduction-115a3d157bad

from sklearn import decomposition, datasets
from sklearn.preprocessing import StandardScaler
features = np.vstack((omn_std, lin_std, plan_std, scat_std, an_std, eig_std, ch_cur_std, mean_el_std, dif_elev_std, r_std, g_std, b_std)).T
features = np.asarray(features)
sc = StandardScaler()
#fit the scaler to the features and transform
features_std = sc.fit_transform(features)
#create a pca object with the 8 components as parameter
pca = decomposition.PCA(n_components=5)
#fit the PCA and transform the data
features_std_pca = pca.fit_transform(features_std)
features_std_pca.shape


plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Features')
plt.ylabel('Variance (%)')
plt.title('Subjective Choose of Features')
plt.show()
