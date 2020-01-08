#################https://towardsdatascience.com/point-cloud-data-simple-approach-f3855fdc08f5    #############
#PCA model
#%% 
#import packages 

import os
import pptk
import numpy as np
from laspy.file import File
import math

#%%

folder = r'C:\Users\laptop\Google Drive\Shared folder Tasos-VanBoven\Sample_data\Broccoli\35m'

las_files = []

for root, dirs, files in os.walk(folder, topdown=True):
    for name in files:
        if name[-4:] == ".las":
            las_files.append(os.path.join(root,name).replace("\\","/"))
            
data_las = File(las_files[39], mode='r') #39 has big differences

xyz = np.vstack([data_las.X, data_las.Y, data_las.Z]).transpose()
rgb = ((np.c_[data_las.Red, data_las.Green, data_las.Blue]) / 255.) / 255.
x = xyz[:,0]
y = xyz[:,1]
z = xyz[:,2]

#remove the median and divide by std = standarlize the data
# =============================================================================
# median = np.median(a=xyz,axis=0)
# std = np.std(a=xyz,axis=0)
# xn = (x - median[0])/std[0]
# yn = (y - median[1])/std[1]
# zn = (z - median[2])/std[2]
# xyz_new = np.vstack([xn,yn,zn]).T
# =============================================================================

#extract the mean from the data
median = np.median(a=xyz,axis=0)
std = np.std(a=xyz,axis=0)
xn = x - median[0]
yn = y - median[1]
zn = z - median[2]
xyz_new = np.vstack([xn,yn,zn]).T
#visualize the data
#v = pptk.viewer(xyz_new,rgb)

#######xyz_new = np.concatenate((xyz_new,rgb),axis = 1) #if we want to put the RGB as well
#%%
#10m resolution of broccoli crops
# =============================================================================
# =============================================================================
# # las_files = []
# # 
# # folder_10m =  r"C:\Users\laptop\Google Drive\Shared folder Tasos-VanBoven\Sample_data\Broccoli\AZ74_10m-1-0 - Cloud.las"
# # for root, dirs, files in os.walk(folder_10m, topdown=True):
# #     for name in files:
# #         if name[-4:] == ".las":
# #             las_files.append(os.path.join(root,name).replace("\\","/"))
# #             data_las = File(las_files[0], mode='r') 
# =============================================================================
# =============================================================================
folder_10m = r"C:\Users\laptop\Google Drive\Shared folder Tasos-VanBoven\Sample_data\Broccoli\AZ74_10m-0-1 - Cloud.las"
data_las = File(folder_10m, mode = 'r')
xyz = np.vstack([data_las.X, data_las.Y, data_las.Z]).transpose()
x = xyz[:,0]
y = xyz[:,1]
z = xyz[:,2]

#remove the median 
median = np.median(a=xyz,axis=0)
xn = x - median[0]
yn = y - median[1]
zn = z - median[2]
xyz_new = np.vstack([xn,yn,zn]).T




#%% 
#Nearest neighbors 



from sklearn.neighbors import NearestNeighbors #import NearestNeigbhors package
nbrs = NearestNeighbors(n_neighbors = 35, algorithm = 'kd_tree').fit(xyz_new) #['auto', 'ball_tree', 'kd_tree', 'brute']
distances, indices = nbrs.kneighbors(xyz_new) #the indices of the nearest neighbors 



#%% 
#pca
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
    coords = xyz_new[(ind),:]
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
# =============================================================================
#     ei = -(e[0]*math.log(e[0])+e[1]*math.log(e[1])+e[2]*math.log(e[2]))
#     eigenentropy.append(ei)
# =============================================================================
    cha = e[2]/sum(e)
    change_curvature.append(cha)
    m_el = z.mean()
    mean_elev.append(m_el)
    d_el = z.max()-z.min()
    dif_elev.append(d_el)
     
    

   
    
#normalization
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
# =============================================================================
# eig = np.asarray(eigenentropy)
# eig_n = (eig -eig.min()) / (eig.max() - eig.min())
# =============================================================================
ch = np.asarray(change_curvature)
ch_cur_n = (ch -ch.min()) / (ch.max() - ch.min())
m_e = np.asarray(mean_elev)
mean_el_n = (m_e -m_e.min()) / (m_e.max() - m_e.min())
d_e = np.asarray(dif_elev)
dif_elev_n = (d_e -d_e.min()) / (d_e.max() - d_e.min())



#visualization
v = pptk.viewer(xyz,lin_n)
v.set(point_size=7)
v.capture('Linearity.png')

v = pptk.viewer(xyz,plan_n)
v.set(point_size=7)
v.capture('Planarity.png')

v = pptk.viewer(xyz,scat_n)
v.set(point_size=7)
v.capture('Scattering.png')

v = pptk.viewer(xyz,an_n)
v.set(point_size=7)
v.capture('Anisotropy.png')

v = pptk.viewer(xyz,eig_n)
v.set(point_size=10)
v.capture('Eigenotropy.png')


v = pptk.viewer(xyz,ch_cur_n)
v.set(point_size=7)
v.capture('Change_of_Curvature.png')

v = pptk.viewer(xyz_new,mean_el_n)
v.set(point_size=7)
v.capture('Mean_elevation.png')

v = pptk.viewer(xyz_new,dif_elev_n)
v.set(point_size=7)
v.capture('Elevation_Difference.png')

v = pptk.viewer(xyz_new,omn_n)
v.set(point_size=7)
v.capture('Omnivariance.png')

# Not performing good when normalize the data with (MinMaxScaler)!!!
    

# =============================================================================
# v = pptk.viewer(xyz,lin_n)
# v.set(point_size=7)
# v.capture('Linearity.png')
# 
# v = pptk.viewer(xyz,plan_n)
# v.set(point_size=7)
# v.capture('Planarity.png')
# 
# v = pptk.viewer(xyz,scat_n)
# v.set(point_size=7)
# v.capture('Scattering.png')
# 
# v = pptk.viewer(xyz,an_n)
# v.set(point_size=7)
# v.capture('Anisotropy.png')
# 
# v = pptk.viewer(xyz,eig_n)
# v.set(point_size=10)
# v.capture('Eigenotropy.png')
# 
# 
# v = pptk.viewer(xyz,ch_cur_n)
# v.set(point_size=7)
# v.capture('Change_of_Curvature.png')
# 
# v = pptk.viewer(xyz,mean_el_n)
# v.set(point_size=1)
# v.capture('Mean_elevation.png')
# 
# v = pptk.viewer(xyz,dif_elev_n)
# v.set(point_size=7)
# v.capture('Elevation_Difference.png')
# 
# v = pptk.viewer(xyz,omn_n)
# v.set(point_size=7)
# v.capture('Omnivariance.png')
# =============================================================================

    

    
    



