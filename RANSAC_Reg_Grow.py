# http://pointclouds.org/documentation/tutorials/region_growing_rgb_segmentation.php
# https://github.com/intel-isl/Open3D/pull/1287
# for images... http://notmatthancock.github.io/2017/10/09/region-growing-wrapping-c.html
# RANSAC and region growing for segmentation
# http://www.hinkali.com/Education/PointCloud.pdf
# https://github.com/strawlab/python-pcl


#%% https://github.com/davidcaron/pclpy/issues/9
# it works for the no_dense point cloud
# region grow
import math
import pclpy
from pclpy import pcl

pc = pclpy.io.las.read( r"C:\Users\laptop\Google Drive\scripts\Pointcloud-processing\no_dense_cloud.las", "PointXYZRGBA")
rg = pcl.segmentation.RegionGrowing.PointXYZRGBA_Normal()
rg.setInputCloud(pc)
normals_estimation = pcl.features.NormalEstimationOMP.PointXYZRGBA_Normal()
normals_estimation.setInputCloud(pc)
normals = pcl.PointCloud.Normal()
normals_estimation.setRadiusSearch(0.01)  #no_dense = 0.01
normals_estimation.compute(normals)
rg.setInputNormals(normals)

rg.setMaxClusterSize(700) #no_dense 700
rg.setMinClusterSize(8) #no_dense 8
rg.setNumberOfNeighbours(4) #no_dense 4
rg.setSmoothnessThreshold(5 / 180 * math.pi)
rg.setCurvatureThreshold(30)
rg.setResidualThreshold(10)
clusters = pcl.vectors.PointIndices()
rg.extract(clusters)
cloud = rg.getColoredCloud()
pclpy.io.las.write(cloud, r"C:\Users\laptop\Google Drive\scripts\Pointcloud-processing/region_growing_no_dense.las")
