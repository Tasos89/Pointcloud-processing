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
4. select points coinside with plant polygons using contains
5. label selected points as 1 for broccoli and include column with plant id
6. export np array as las file.

"""

import os
import numpy as np
import geopandas as gpd
import pandas as pd
from laspy.file import File
import laspy.header


#plant shape path
input_plant_path = r"D:\New folder\Nieuwe map\merged_shapes.shp"
#pointcloud folder



folder = r'G:\Shared drives\400 Development\Student Assignments\Interns\Tasos\Sample_data\Broccoli\wgs84'

output_folder = r'G:\Shared drives\400 Development\Student Assignments\Interns\Tasos\Sample_data\Broccoli'
#read files
plant_shapes = gpd.read_file(input_plant_path)


las_files = []

for root, dirs, files in os.walk(folder, topdown=True):
    for name in files:
        if name[-4:] == ".las":
            las_files.append(os.path.join(root,name).replace("\\","/"))

las_list = []
for file in las_files[:5]:
    data_las = File(file, mode='r') #39 has big differences
    xyz = np.vstack([data_las.x, data_las.y, data_las.z]).transpose()
    las_list.append(xyz)

#merge to one array    
xyz = np.concatenate(las_list, axis = 0)

#convert array into dataframe
df = pd.DataFrame(xyz, columns=['X','Y','Z'])

#convert to geodataframe # https://geopandas.readthedocs.io/en/latest/gallery/create_geopandas_from_pandas.html
gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.X, df.Y))

#Initialize the reference system of the points
#gdf.crs = {'init' :'epsg:28992'}
gdf.crs = ({'init' :'epsg:4326'})
#gdf = gdf.to_crs({'init' :'epsg:28992'})

gdf['plant_id'] = 0

#Coinside of points and polygons shapes
#for i, plant in enumerate(plant_shapes):
contained = plant_shapes.geometry.contains(gdf.geometry)
contained2 = gdf.geometry.contains(plant_shapes.geometry)
#gdf.plant_id.loc[contained[contained == True].index] = i+1    
test = gpd.sjoin(gdf, plant_shapes.iloc[:50], how = 'inner', op = 'intersects')
test = gpd.sjoin(plant_shapes.iloc[50:100], gdf, how = 'inner', op = 'intersects')


gdf.to_file(r'd:\points.shp')
plant_shapes.to_file(r'd:\shapes.shp')

#convert gdf to array
broccoli_points = gdf[].to_numpy() #individual broccoli points labeled ~= 0


#save array
np.save('broccoli_points', broccoli_points)

#gdf.iloc[:].to_file(output_folder + r'\points_broccoli.gpkg')

#convert contained broccoli points from geodataframe to .las and save it
outfile = laspy.file.File(r'G:\Shared drives\400 Development\Student Assignments\Interns\Tasos\Sample_data\Broccoli\broccoli.las', mode="w", header=header)
outfile.X = contained[:,0]
outfile.Y = contained[:,1]
outfile.Z = contained[:,2]
outfile.Red = contained[:,3]
outfile.Greeen = contained[:,4]
outfile.Blue = contained[:,5]
outfile.colse()
