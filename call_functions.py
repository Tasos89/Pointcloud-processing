#%%main scricpt
import os
from laspy.file import File
import sys
sys.path.append(r"C:\Users\laptop\Google Drive\scripts")


folder_35m = r'C:\Users\laptop\Google Drive\Shared folder Tasos-VanBoven\Sample_data\Broccoli\35m' #39 .las file is interesting
folder_10m =  r'C:\Users\laptop\Google Drive\Shared folder Tasos-VanBoven\Sample_data\Broccoli'
folder_Brussels = r'C:\Users\laptop\Google Drive\Shared folder Tasos-VanBoven\Sample_data\Brussel sprouts'



#Read las files - returns data_las

#%%
import pptk_visualization_las
data_las = pptk_visualization_las.read_las(folder_10m)



#visualize las files - with the enrichenrichment of RGB



xyz, rgb = pptk_visualization_las.visualize_las(data_las) #return xyz, rgb



#1rst filter: extract the mean height from the z-values and filter delete the outliers


#%%
import filtering_height_rgb
xyz_fil1, rgb_fil1 = filtering_height_rgb.filter1(xyz, data_las, rgb)


#%%
#2nd filter: extract some ground values based on color 



xyz_fil2, rgb_fil2 = filtering_height_rgb.filter2(xyz_fil1, rgb_fil1)


#%%
#3rd filter: delete some ground values based on color (extreme case)



xyz_fil3, rgb_fil3 = filtering_height_rgb.filter3(xyz_fil2,rgb_fil2)



#Knn nearest neigbors 


#%%
import nearest_pca
indices = nearest_pca.NN(xyz)



#pca



li_norm, plan_norm, scat_norm, omn_norm, anis_norm, cha_curve_norm, latent = nearest_pca.pca(xyz,indices)

