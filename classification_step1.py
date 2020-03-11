?#%% import packages 
import os
import pptk
import numpy as np
from laspy.file import File
from skimage.color import rgb2lab
from sklearn.neighbors import NearestNeighbors 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import time
import math

#%%

#specifiy the folder/file for the .las files. 
#path = r"C:\Users\laptop\Google Drive\Shared folder Tasos-VanBoven\Sample_data\Broccoli\AZ74_10m-0-1 - Cloud.las"
#path = r"C:\Users\laptop\Google Drive\Shared folder Tasos-VanBoven\Sample_data\Broccoli\35m\Rijweg_stalling1-9-5.las"
path = r"C:\Users\laptop\Google Drive\Shared folder Tasos-VanBoven\Sample_data\ultra_high_res-1-1 - Cloud.las"

if path.endswith('las'):
    data_las = File(path, mode = 'r')
    #extract the coordinates of the .las file
    xyz = np.vstack([data_las.x, data_las.y, data_las.z]).transpose()
    rgb = ((np.c_[data_las.Red, data_las.Green, data_las.Blue]) / 255.) / 255. #normalized
else:
    las_files = []
    for root, dirs, files in os.walk(path, topdown=True):
        for name in files:
            if name[-4:] == ".las": 
                las_files.append(os.path.join(root,name).replace("\\","/"))
    las_list = []
    rgb_l = []
    for file in las_files[:10]: #specify the number of file you want to open
        data_las = File(file, mode='r')
        rgb = ((np.c_[data_las.Red, data_las.Green, data_las.Blue]) / 255.) / 255. #normalized
        xyz = np.vstack([data_las.x, data_las.y, data_las.z]).transpose()
        las_list.append(xyz)
        rgb_l.append(rgb)
    #extract the coordinates of the .las file
    xyz = np.concatenate(las_list, axis = 0)
    rgb = np.concatenate(rgb_l, axis = 0)

    

#extract the color of the .las file
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

# from BGR to Lab # https://gist.github.com/bikz05/6fd21c812ef6ebac66e1
# The results using Lab instead of RGB was not so good, hence changing the color space didn't work for early crop data
#def func(t):
#    if (t > 0.008856):
#        return np.power(t, 1/3.0);
#    else:
#        return 7.787 * t + 16 / 116.0;
#
##Conversion Matrix
#matrix = [[0.412453, 0.357580, 0.180423],
#          [0.212671, 0.715160, 0.072169],
#          [0.019334, 0.119193, 0.950227]]
#
## RGB values lie between 0 to 1.0
#Lab_OpenCv = []
#Lab = np.zeros((len(rgb),3))
#for row in rgb:
#    cie = np.dot(matrix, row);
#    
#    cie[0] = cie[0] /0.950456;
#    cie[2] = cie[2] /1.088754; 
#    
#    # Calculate the L
#    L = 116 * np.power(cie[1], 1/3.0) - 16.0 if cie[1] > 0.008856 else 903.3 * cie[1];
#    
#    # Calculate the a 
#    a = 500*(func(cie[0]) - func(cie[1]));
#    
#    # Calculate the b
#    b = 200*(func(cie[1]) - func(cie[2]));
#    
#    #  Values lie between -128 < b <= 127, -128 < a <= 127, 0 <= L <= 100 
#    Lab = [b , a, L]; 
#    
#    # OpenCV Format
#    L = L * 255 / 100;
#    a = a + 128;
#    b = b + 128;
#    Lab_OpenCv.append([b,a,L])
##Lab_OpenCv = np.asarray(Lab_OpenCv)
##scaler = MinMaxScaler()
##Lab = scaler.fit_transform(Lab_OpenCv)
##Lab[:,2] = 0.0
##b = Lab[:,0]
##a = Lab[:,1]
##L = Lab[:,2]
#Lab = np.asarray(Lab_OpenCv)
#b = (Lab[:,0] - Lab[:,0].min()) / (Lab[:,0].max() - Lab[:,0].min())
#a = (Lab[:,1] - Lab[:,1].min()) / (Lab[:,1].max() - Lab[:,1].min())
#L = (Lab[:,2] - Lab[:,2].min()) / (Lab[:,2].max() - Lab[:,2].min())
#Lab = np.vstack((b,a,L)).T

v = pptk.viewer(xyz,rgb)
v.set(point_size = 0.01)

#%% 
# Nearest neighbors with normalized data. The confusion matrix return better results but the visualization was not so good
start_time = time.clock()

nbrs = NearestNeighbors(n_neighbors = 35, algorithm = 'kd_tree').fit(xyz_nn) #['auto', 'ball_tree', 'kd_tree', 'brute']
distances, indices = nbrs.kneighbors(xyz_nn) #the indices of the nearest neighbors 



# extraction of geometrical features among the nearest neighbors 
# https://towardsdatascience.com/an-approach-to-choosing-the-number-of-components-in-a-principal-component-analysis-pca-3b9f3d6e73fe


linearity = []
planarity = []
scatter = []
omnivariance = []
anisotropy = []
change_curvature = []
dif_elev = []
mean_elev = []
omnivariance = []
eigenentropy = []

for i in range(len(indices)):
    ind = indices[i]
    coords = xyz_nn[(ind),:] # for normalize data
    #coords = xyz_std[(ind),:] #for standarlize

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
    ei = -(e[0]*math.log(e[0])+e[1]*math.log(e[1])+e[2]*math.log(e[2]))
    eigenentropy.append(ei)
     
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
ch = np.asarray(change_curvature)
ch_cur_n = (ch -ch.min()) / (ch.max() - ch.min())
m_e = np.asarray(mean_elev)
mean_el_n = (m_e -m_e.min()) / (m_e.max() - m_e.min())
d_e = np.asarray(dif_elev)
dif_elev_n = (d_e -d_e.min()) / (d_e.max() - d_e.min())
eig = np.asarray(eigenentropy)
ei_n = (eig -eig.min()) / (eig.max() - eig.min())

print (time.clock() - start_time, "seconds")

#visualization
# =============================================================================
# v = pptk.viewer(xyz,lin_n)
# v.set(point_size=0.02)
# 
# v = pptk.viewer(xyz,plan_n)
# v.set(point_size=0.01)
# 
# v = pptk.viewer(xyz,scat_n)
# v.set(point_size=0.02)
# 
# v = pptk.viewer(xyz,an_n)
# v.set(point_size=0.02)
# 
# v = pptk.viewer(xyz,ch_cur_n)
# v.set(point_size=0.02)
# 
# v = pptk.viewer(xyz,mean_el_n)
# v.set(point_size=0.02)
# 
# v = pptk.viewer(xyz,dif_elev_n)
# v.set(point_size=0.01)
# 
# v = pptk.viewer(xyz,omn_n)
# v.set(point_size=0.01)
# 
# v = pptk.viewer(xyz,ei_n)
# v.set(point_size=0.02)
# =============================================================================


# features chooce
if 'Lab' in locals():
    features = np.vstack((omn_n, dif_elev_n, ei_n, a, b)).T #Lab
else:
    features = np.vstack((omn_n, dif_elev_n, R, G, B)).T #rgb

#%% PCA
# pca for the geometrical features to define the most important features with minmaxscaler features
# https://www.visiondummy.com/2014/05/feature-extraction-using-pca/
# https://chrisalbon.com/machine_learning/feature_engineering/feature_extraction_with_pca/ 
# https://towardsdatascience.com/an-approach-to-choosing-the-number-of-components-in-a-principal-component-analysis-pca-3b9f3d6e73fe

#features = np.vstack((omn_n, lin_n, plan_n, scat_n, an_n, ch_cur_n, mean_el_n, dif_elev_n, L, a, b)).T #Lab
#features = np.vstack((omn_n, lin_n, plan_n, scat_n, an_n, ch_cur_n, mean_el_n, dif_elev_n, R, G, B)).T #rgb

pca = PCA().fit(features)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Features')
plt.ylabel('Variance (%)')
plt.title('Subjective Choose of Features')
plt.show()

# Hence we can keep 5 attributes/features that have the variance of almost all the data ~100%
pca1 = PCA(n_components=5)
features_new = pca1.fit_transform(features)