#%% from point clouds to 2D imaginary processing

#%% libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
#from scipy.interpolate import griddata

#%% read data
#input_path = r'C:\Users\laptop\Google Drive\scripts\Pointcloud-processing\out_file.csv'
input_path = r"C:\Users\laptop\Google Drive\scripts\Pointcloud-processing\dense_cloud_plants.csv"
data = pd.read_csv(input_path)
data = np.asarray(data)

#%% from 3D to 2D
N = 1000
xstart = np.min(data[:,0])
ystart = np.min(data[:,1])
xstep = (np.max(data[:,0])-np.min(data[:,0]))/N
ystep = (np.max(data[:,1])-np.min(data[:,1]))/N
im = np.zeros((N,N))
for k in range(len(data)):
    i = int((data[k,0]-xstart)/xstep) -1
    j = int((data[k,1]-ystart)/ystep) -1
    #im[i,j]=data[k,2]*100000
    im[i,j] = 1
plt.figure()
plt.imshow(im)

#%% save numpy array to .png 

#matplotlib.image.imsave(r'C:\Users\laptop\Google Drive\pictures for the internship report\2D2.png', im)
matplotlib.image.imsave(r'C:\Users\laptop\Google Drive\pictures for the internship report\2D_dense.png', im)


#%% read it as a grey scale image

#img = cv2.imread(r'C:\Users\laptop\Google Drive\pictures for the internship report\2D2.png')
img = cv2.imread(r'C:\Users\laptop\Google Drive\pictures for the internship report\2D_dense.png')
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('Gray image', grey)

grey = np.asarray(grey).astype(float)

int = np.where(grey!=30)
grey[int] = None

plt.imshow(grey)

matplotlib.image.imsave(r'C:\Users\laptop\Google Drive\pictures for the internship report\2D_dense.png', grey)
