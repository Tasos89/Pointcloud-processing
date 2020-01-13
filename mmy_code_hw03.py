# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 11:35:39 2019

@author: timo1
"""
# Q for friday: what to do with point outside initial TIN
import math

# for reading LAS files
from laspy.file import File
import numpy as np

# triangulation for ground filtering algorithm and TIN interpolation 
import startin
import pptk
# kdtree for IDW interpolation
from scipy.spatial import cKDTree
import json, sys
import matplotlib as plt


jparams = json.load(open('params.json'))
thinning_factor = jparams["thinning-factor"]



inputfile = File('D:/Documents/Studie/DTM/hw03/ahn3/rural.las',mode="r")
Head = inputfile.header
Offset = Head.offset
Scale = Head.scale
cloud = inputfile.points
x = inputfile.x
y = inputfile.y
z = inputfile.z

thin = np.arange(0,len(x),thinning_factor)
x = x[thin]
y = y[thin]
z = z[thin]

gf_cellsize = jparams["gf-cellsize"]

#create a grid
xmin = np.floor(np.min(x))
ymin = np.floor(np.min(y))
xmax = np.ceil(np.max(x))
ymax = np.ceil(np.max(y))
x_grid = np.arange(xmin,xmax,gf_cellsize)
y_grid = np.arange(ymin,ymax,gf_cellsize)

ground_points = []

for i in np.arange(0,len(x_grid)-1):
    for j in np.arange(0,len(y_grid)-1):  
        points_cell = np.where((x > x_grid[i]) & (y>y_grid[j]) & (x<x_grid[i+1]) & (y<y_grid[j+1]))
        z_points = z[points_cell[0]]
        point_lowest = points_cell[0][np.argmin(z_points)]
        ground_points.append(point_lowest)
        
x_lowest = x[ground_points]
y_lowest = y[ground_points]
z_lowest = z[ground_points]
x = np.delete(x,ground_points)
y = np.delete(y,ground_points)
z = np.delete(z,ground_points)
xyz_lowest = np.vstack((x_lowest,y_lowest,z_lowest)).T
dt = startin.DT()
dt.insert(xyz_lowest)


points_outside = []

for i in np.arange(21,22):
    Triangle = dt.locate(x[i],y[i])
    x_point = x[i]
    y_point = y[i]
    z_point = z[i]
    
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
        x_point = x_point-Xm
        y_point = y_point-Ym
        z_point = z_point-Zm
        xyz_point = np.array([x_point,y_point,z_point]).reshape(-1,1).T
        x_tri = (x_tri-Xm).reshape(-1,1)
        y_tri = (y_tri-Ym).reshape(-1,1)
        z_tri = (z_tri-Zm).reshape(-1,1)
        M = np.concatenate((x_tri,y_tri,z_tri),axis=1)
        Mt = M.transpose()
        MtM = np.dot(Mt,M)
        e, v = np.linalg.eig(MtM)
        emax = np.argmax(e)
        emin = np.argmin(e)
        emid = 3-emax-emin
        v2 = np.zeros([3,3])
        v2[:,0] = v[:,emax]
        v2[:,1] = v[:,emid]
        v2[:,2] = v[:,emin]
        regionbase = np.dot(v2.T,M.T)
        point_new = np.dot(v2.T,xyz_point.T)

        
    else:
        points_outside.append(i)
        
        

x_outside = x[points_outside]
y_outside = y[points_outside]
z_outside = z[points_outside]
xyz = np.vstack((x_point,y_point,z_point)).T
# =============================================================================
# xyz_outside = np.vstack((x_outside,y_outside,z_outside)).T
# v = pptk.viewer(xyz_outside)
# v.set(point_size=1)
# =============================================================================
Expires in 3 weeks, 6 daysNumPy
API | About | Plugins