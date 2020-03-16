# Ground filtering usning Delaunay Triangulation
import laspy
import math
from laspy.file import File
import numpy as np
import startin
import pptk
import json, sys

#%% Read the data / initialize the parameters

input_las =  "your_path.las"
inputfile = File(input_las,mode="r")
x = inputfile.x
y = inputfile.y
z = inputfile.z
xyz = np.vstack((x,y,z)).T

#%%initialize the parameters 

gf_distance = 0.008 # vertical distance in [m]
gf_angle = 10 # the maximum angle of the edges in [degrees]
gf_cellsize = 1 # the size of the grid edge in [m]    

#%% Extract the lower point of every cell

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
        if points_cell[0].shape[0]>0:
            z_points = z[points_cell[0]]
            point_lowest = points_cell[0][np.argmin(z_points)]
            ground_points.append(point_lowest)
        
x_lowest = x[ground_points]
y_lowest = y[ground_points]
z_lowest = z[ground_points]
xyz_lowest = np.vstack((x_lowest,y_lowest,z_lowest)).T

x = np.delete(x,ground_points)
y = np.delete(y,ground_points)
z = np.delete(z,ground_points)
remaining = np.concatenate((x.reshape(-1,1),y.reshape(-1,1),z.reshape(-1,1)),axis=1)

#%% Creation of the DTM - DSM

#insert 4 extra point to create a convex hull that includes all points in the dataset
xyz_outside = np.array([[xmin-gf_cellsize,ymin-gf_cellsize,z_lowest[0]],[xmin-gf_cellsize,ymax+gf_cellsize,z_lowest[len(y_grid)-2]],[xmax+gf_cellsize,ymin-gf_cellsize,z_lowest[len(y_lowest)-(len(y_grid)-1)]],[xmax+gf_cellsize,ymax+gf_cellsize,z_lowest[len(y_lowest)-1]]])
points_initial = np.concatenate((xyz_outside,xyz_lowest),axis=0)
#added = points_initial.shape[0]

dt = startin.DT()
dt.insert(points_initial)

points_outside = []
k = 0
check = 0

while remaining.shape[0]>k:
    print(remaining.shape[0]-k)

    
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
        if distance<gf_distance and alpha<gf_angle: #distance 0.009 was nice alpha =8
            dt.insert_one_pt(x_point0,y_point0,z_point0)
            remaining = np.delete(remaining,k,0)
            check = 1
            
        else:
            k = k+1
    else:
        points_outside.append(k)
        k = k+1
    if check==1 and k==remaining.shape[0]:
        k=0
        check = 0
      
vertices = dt.all_vertices()
vertices = np.asarray(vertices)
#delete all vertices with strange values 
vertices = np.delete(vertices,0,0)
vertices = np.delete(vertices,[4,5,119,122,127],0)
#delete the corner points that were added:
vertices = np.delete(vertices,[0,1,2,3],0)
#adjust the indices of the triangles for the deleted vertices:
triangles = np.asarray(dt.all_triangles())-10

DSM = np.array(remaining)
DTM = np.array(vertices)

#%% visualization of dtm-dsm

v = pptk.viewer(DSM)
v.set(point_size=0.01)
v = pptk.viewer(DTM)
v.set(point_size=0.01)

#%% write the data as .csv and .obj
dtm = np.savetxt(your_path_and_name.csv,DTM)
dsm = np.savetxt(your_path_and_name.csv,DSM)
dt.write_obj(your_path_and_name.obj) #for view on CloudCompare 
