#  counting features from 2D image
#%% import libraries

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import feature, color

#%% reade the 2D images preferably in grey scale

input_path = r'C:\Users\laptop\Google Drive\pictures for the internship report\2D.png'
#input_path = r'C:\Users\laptop\Google Drive\pictures for the internship report\2D_dense.png'
img = mpimg.imread(input_path)
plt.imshow(img)
plt.show()


#%% for early stage broccoli

bw = img.mean(axis=2)

fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(1,1,1)

blobs_dog = [(x[0],x[1],x[2]) for x in feature.blob_dog(-bw, 
                                                        min_sigma=8, 
                                                        max_sigma=15, 
                                                        threshold=0.004, 
                                                        overlap=0.002)] 

#remove duplicates
blobs_dog = set(blobs_dog)

img_blobs = color.gray2rgb(img)

for blob in blobs_dog:
    y, x, r = blob
    c = plt.Circle((x, y), r+1, color='red', linewidth=2, fill=False)
    ax.add_patch(c)

plt.imshow(img_blobs)
plt.title('Center of broccoli plants')

plt.show()
print('Number of center broccoli plants detected: ' + str(len(blobs_dog)))

#%% for harvesting time broccoli

bw = img.mean(axis=2)

fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(1,1,1)

blobs_dog = [(x[0],x[1],x[2]) for x in feature.blob_dog(-bw, 
                                                        min_sigma=90, 
                                                        max_sigma=100, 
                                                        threshold=0.4, 
                                                        overlap=0.002)] 

#remove duplicates
blobs_dog = set(blobs_dog)

img_blobs = color.gray2rgb(img)

for blob in blobs_dog:
    y, x, r = blob
    c = plt.Circle((x, y), r+1, color='red', linewidth=2, fill=False)
    ax.add_patch(c)

plt.imshow(img_blobs)
plt.title('Center of broccoli plants')

plt.show()
print('Number of center broccoli plants detected: ' + str(len(blobs_dog)))

