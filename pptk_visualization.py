#%% 
#import packages
import os
import pptk
import plyfile
import numpy as np
from laspy.file import File

#covert 3D (.ply) into mesh (with faces and verices)
import trimesh
from scipy.spatial import Delaunay
import open3d as o3d

#%%
#Directories/roots 
#read the files
folder = r'C:\Users\laptop\Google Drive\Shared folder Tasos-VanBoven\Sample_data\Broccoli\35m'


ply_files = []
for root, dirs, files in os.walk(folder, topdown=True):
    for name in files:
        if name[-4:] == ".ply":
            ply_files.append(os.path.join(root,name).replace("\\","/"))
            
for file in ply_files:
    data_ply = plyfile.PlyData.read(file)['vertex'] #read the data

    
las_files = []
for root, dirs, files in os.walk(folder, topdown=True):
    for name in files:
        if name[-4:] == ".las":
            las_files.append(os.path.join(root,name).replace("\\","/"))
            
data_las = File(las_files[55], mode='r')

#%%
#reading mesh data 
from pyntcloud import PyntCloud
data_plyy = PyntCloud.from_file(r"C:\Users\laptop\Google Drive\Shared folder Tasos-VanBoven\Sample_data\Broccoli\35m\meshed.ply") #gives different features like\
#number of point and faces. But i cannot visualize it.
data_ply = plyfile.PlyData.read(r"C:\Users\laptop\Google Drive\Shared folder Tasos-VanBoven\Sample_data\Broccoli\35m\meshed.ply")['vertex']

#%%
#procesing/visualization of ply point clouds data

xyz = np.c_[data_ply['x'], data_ply['y'], data_ply['z']]
rgb = np.c_[data_ply['red'], data_ply['green'], data_ply['blue']]
n = np.c_[data_ply['nx'], data_ply['ny'], data_ply['nz']]
v = pptk.viewer(xyz)
v.attributes(rgb / 255., 0.5 * (1 + n))
v.set(point_size=0.01)

#%%
#procesing/visualization of ply meshe



#%% 
#preocessing/visualization of las data

#rgb = np.c_[data_las['red'], data_las['green'], data_las['blue']]
#n = np.c_[data_las['nx'], data_las['ny'], data_las['nz']]
data_las = data_las.points
xyz = np.c_[data_las['x'], data_las['y'], data_las['z']]
rgb = np.c_[data_las['red'], data_las['green'], data_las['blue']]
n = np.c_[data_las['nx'], data_las'ny'], data_las['nz']]

#lasPoints = clasiffication == 2
v = pptk.viewer(data_las)
#v.attributes(rgb / 255., 0.5 * (1 + n))

#%%
def read_pointcloud(input_file):
    
    
    
    return point_cloud_arr



#%%
