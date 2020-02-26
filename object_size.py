#%%
import cv2
import numpy as np 
 
broccoli	= cv2.imread(r'C:\Users\laptop\Google Drive\pictures for the internship report\2D.png')
gray_img	=	cv2.cvtColor(broccoli,	cv2.COLOR_BGR2GRAY)
img	= cv2.medianBlur(gray_img,	5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
 
#center
 
circles	= cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,120,param1=100,param2=30,minRadius=0,maxRadius=0)
circles	= np.uint16(np.around(circles))
 
for	i in circles[0,:]:
				#	draw	the	outer	circle
				cv2.circle(broccoli,(i[0],i[1]),i[2],(0,255,0),6)
				#	draw	the	center	of	the	circle
				cv2.circle(broccoli,(i[0],i[1]),2,(0,0,255),3)
 
cv2.imshow("HoughCirlces",	broccoli)
cv2.waitKey()
cv2.destroyAllWindows()

#%%

from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
import cv2
import matplotlib.pyplot as plt


image	= cv2.imread(r'C:\Users\laptop\Google Drive\pictures for the internship report\2D.png')
image_gray = rgb2gray(image)

blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=1, threshold=.1)

# Compute radii in the 3rd column.
blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=.1)
blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=.01)

blobs_list = [blobs_log, blobs_dog, blobs_doh]
colors = ['yellow', 'lime', 'red']
titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
          'Determinant of Hessian']
sequence = zip(blobs_list, colors, titles)

fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

for idx, (blobs, color, title) in enumerate(sequence):
    ax[idx].set_title(title)
    ax[idx].imshow(image)
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
        ax[idx].add_patch(c)
    ax[idx].set_axis_off()

plt.tight_layout()
plt.show()

#%%
#%% import libraries

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import feature, color

#%% reade the 2D images

input_path = r'C:\Users\laptop\Google Drive\pictures for the internship report\2D.png'
#input_path = r'C:\Users\laptop\Google Drive\pictures for the internship report\2D_dense.png'
image = mpimg.imread(input_path)
plt.imshow(img)
plt.show()


#%% for early stage broccoli
from skimage.color import rgb2gray
image_gray = rgb2gray(img)

from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray

import matplotlib.pyplot as plt



from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray

import matplotlib.pyplot as plt


blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=.01)

blobs_list = [blobs_doh]
colors = ['yellow', 'lime', 'red']
titles = ['Determinant of Hessian']
sequence = zip(blobs_list, colors, titles)

fig, axes = plt.subplots(1, 1, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

ax.set_title(title)
ax.imshow(image)

y, x, r = blobs
c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
ax.add_patch(c)
ax.set_axis_off()

plt.tight_layout()
plt.show()
