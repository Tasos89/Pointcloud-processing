#%% from point clouds to 2D imaginary processing

#%% libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

#%% read data
data = pd.read_csv(r'C:\Users\laptop\Google Drive\scripts\Pointcloud-processing\out_file.csv')
data = np.asarray(data)
x = data[:,0]
y = data[:,1]
z = data[:,2]
xy = np.vstack((x,y)).T

#%% interpolation

x_range=(np.max(x)-np.min(x)
y_range=(np.max(y)-np.min(y)
grid_x, grid_y = np.mgrid[np.min(x):np.max(x):(x_range*1j), np,min(y):np.max(y):(y_range*1j)]
points = df[['X','Y']].values
values = df['new'].values
grid_z0 = griddata(points, values, (grid_x, grid_y), method='linear').astype(np.uint8)
im=Image.fromarray(grid_z0,'L')
im.show()

#%% Image
# normalize the data and convert to uint8 (grayscale conventions)
zNorm = (z - np.min(z)) / (np.max(z) - np.min(z)) * 255
zNormUint8 = zNorm.astype(np.uint8)

# plot result
plt.figure()
plt.imshow(zNormUint8)
