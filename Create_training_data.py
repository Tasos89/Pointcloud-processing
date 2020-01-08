# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:18:26 2020

@author: ericv
"""

"""
ToDo:
    
1. read plant shapes in geodataframe 
2. read point cloud in numpy array
3. convert pointcloud to geodataframe
4. select points in plant polygons using contains
5. label selected points as 1 for broccoli and include column with plant id
6. export np array as las file.

"""

import os
import numpy as np
import geopandas as gpd
import pandas as pd
from laspy.file import File

#plant shape path
#input_plant_path = 
#pointcloud folder
folder = r'G:\Gedeelde drives\400 Development\Student Assignments\Interns\Tasos\Sample_data\Broccoli\35m'

output_folder = r'C:\Users\ericv\Pictures\please_sent_to_Tasos'
#read files
#plant_shapes = gpd.read_file(input_plant_path)


las_files = []

for root, dirs, files in os.walk(folder, topdown=True):
    for name in files:
        if name[-4:] == ".las":
            las_files.append(os.path.join(root,name).replace("\\","/"))

las_list = []
for file in las_files[:]:
    data_las = File(file, mode='r') #39 has big differences
    xyz = np.vstack([data_las.x, data_las.y, data_las.z]).transpose()
    las_list.append(xyz)

#merge to one array    
xyz = np.concatenate(las_list, axis = 0)

#convert array into dataframe
df = pd.DataFrame(xyz, columns=['X','Y', 'Z'])

#convert to geodataframe
gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.X, df.Y))

gdf.crs = {'init' :'epsg:28992'}

gdf['plant_id'] = 0

for i, plant in enumerate(plant_shapes):
    contained = gpd.contains(plant, gdf)
    gdf.plant_id.loc[contained[contained == True].index] = i+1    

#convert gdf to array

#save array
np.save()

#gdf.iloc[:].to_file(output_folder + r'\points_broccoli.gpkg')

