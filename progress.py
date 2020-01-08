#%%
#import packages




import os
import pptk
import numpy as np
from laspy.file import File
import whitebox_tools as wbt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt




#%%
#reading las files



folder = r'C:\Users\laptop\Google Drive\Shared folder Tasos-VanBoven\Sample_data\Broccoli\35m'
las_files = []

for root, dirs, files in os.walk(folder, topdown=True):
    for name in files:
        if name[-4:] == ".las":
            las_files.append(os.path.join(root,name).replace("\\","/"))
            
data_las = File(las_files[39], mode='r')
    



#%%
#visualize las files




xyz = np.vstack([data_las.X, data_las.Y, data_las.Z]).transpose()
rgb = ((np.c_[data_las.Red, data_las.Green, data_las.Blue]) / 255.) / 255.

v = pptk.viewer(xyz,rgb)
v.set(point_size=10)
#v.capture('screenshot1.png')

#%% 1rst filtering
#preprocess the data (erase the mean heigh value [units?] - filtering of the outliers (canals, hils, etc..)




z_mean = xyz[:,2].mean() #mean value of raw data height [units?]..maybe [mm?]
z_new = (xyz[:,2] - z_mean) #extract the mean heigh value 
sd = z_new.std()
z_mean = z_new.mean()
xyz_new = np.vstack([data_las.X, data_las.Y, z_new]).transpose()

delete_indexes = []
for i in range(len(xyz)):
    if z_new[i] > sd or z_new[i] < - 0.00001 *sd:
                delete_indexes.append(i)

xyz_fil = np.delete(xyz_new, delete_indexes, 0)
rgb_fil = np.delete(rgb, delete_indexes, 0)

v_new = pptk.viewer(xyz_fil,rgb_fil)
v_new.set(point_size=10)




#%% 2nd filtering 
#Delete the data with R - 0, G - 0,B - 0 / when B == 0 should be ground

#index_rgb = rgb_fil.index(rgb_fil[:,2] == 0)
#index_rgb_zero, = rgb_fil.where(rgb_fil[:.2] == 0)
index_zero = np.where(rgb_fil[:,2] < 0.6)
xyz_nonzero = np.delete(xyz_fil, index_zero, 0)
rgb_nonzero = np.delete(rgb_fil, index_zero, 0)

v_nonzero = pptk.viewer(xyz_nonzero, rgb_nonzero)
v_nonzero.set(point_size=10)

# =============================================================================
# index_ = np.where(rgb_nonzero[:,0]  > 0.8)
# xyz_ = np.delete(xyz_nonzero, index_, 0)
# rgb_ = np.delete(rgb_nonzero, index_, 0)
# 
# v_ = pptk.viewer(xyz_, rgb_)
# v_.set(point_size=10)
# =============================================================================

#delete the non green points ||
index_green = np.where(rgb_nonzero[:,1] <0.9)
rgb_green = np.delete(rgb_nonzero, index_green, 0)
xyz_green = np.delete(xyz_nonzero, index_green, 0)

v_green = pptk.viewer(xyz_green, rgb_green)
v_green.set(point_size=10)



    




#%%
#delaunay triangulation 

# =============================================================================
# from scipy.spatial import Delaunay
# tri = Delaunay(xyz_fil)
# vertices = tri.vertices
# points = tri.points
# 
# edge_idx = np.unique(tri.convex_hull)
# 
# =============================================================================


from scipy.interpolate import griddata

# Create some test data, 3D gaussian, 200 points
dx, pts = 2, 100j

N = 500
R = np.random.random((N,3))*2*dx - dx
V = np.exp(-( (R**2).sum(axis=1)) )

# Create the grid to interpolate on
X,Y,Z = np.mgrid[-dx:dx:pts, -dx:dx:pts, -dx:dx:pts]

# Interpolate the data
F = griddata(R, V, (X,Y,Z))

from mayavi.mlab import *
contour3d(F,contours=8,opacity=.2 )

VV = np.exp(-((xyz_fil**2).sum(axis=1)))
FF = griddata(xyz_fil, VV, (X,Y,Z))
