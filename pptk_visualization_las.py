#%%
#import packages




import os
import pptk
import numpy as np
from laspy.file import File
# =============================================================================
# import whitebox_tools as wbt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# 
# =============================================================================

#%%
#reading las files



#folder = r'C:\Users\laptop\Google Drive\Shared folder Tasos-VanBoven\Sample_data\Broccoli\35m'

def read_las(folder):
    las_files = []
    
    for root, dirs, files in os.walk(folder, topdown=True):
        for name in files:
            if name[-4:] == ".las":
                las_files.append(os.path.join(root,name).replace("\\","/"))
                
    data_las = File(las_files[0], mode='r')
    return data_las
    



#%%
#visualize las files



def visualize_las(data_las):
    xyz = np.vstack([data_las.X, data_las.Y, data_las.Z]).transpose()
    rgb = ((np.c_[data_las.Red, data_las.Green, data_las.Blue]) / 255.) / 255.
    
    v = pptk.viewer(xyz,rgb)
    v.set(point_size=10)
    #v.capture('screenshot1.png')
    return xyz, rgb

#%%
#preprocess the data (erase the mean heigh value [units?] - filtering of the outliers (canals, hils, etc..)




# =============================================================================
# z_mean = xyz[:,2].mean() #mean value of raw data height [units?]..maybe [mm?]
# z_new = (xyz[:,2] - z_mean) #extract the mean heigh value 
# sd = z_new.std()
# z_mean = z_new.mean()
# xyz_new = np.vstack([data_las.X, data_las.Y, z_new]).transpose()
# 
# delete_indexes = []
# for i in range(len(xyz)):
#     if z_new[i] > sd or z_new[i] < - 0.00001 *sd:
#                 delete_indexes.append(i)
# 
# xyz_fil = np.delete(xyz_new, delete_indexes, 0)
# rgb_fil = np.delete(rgb, delete_indexes, 0)
# 
# v_new = pptk.viewer(xyz_fil,rgb_fil)
# v_new.set(point_size=10)
# =============================================================================

