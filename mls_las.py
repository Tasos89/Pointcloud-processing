#%%

import pclpy
from pclpy import pcl

#%%

point_cloud = pclpy.read( r"C:\Users\laptop\Google Drive\Google Drive\Shared folder Tasos-VanBoven\Sample_data\Broccoli\35m\Rijweg_stalling1-8-5.las", "PointXYZRGBA")
mls = pcl.surface.MovingLeastSquaresOMP.PointXYZRGBA_PointNormal()
tree = pcl.search.KdTree.PointXYZRGBA()
mls.setSearchRadius(0.05)
mls.setPolynomialFit(False)
mls.setNumberOfThreads(12)
mls.setInputCloud(point_cloud)
mls.setSearchMethod(tree)
mls.setComputeNormals(True)
output = pcl.PointCloud.PointNormal()
mls.process(output)



# =============================================================================
# visual = pclpy.show(tree)
# visual.ShowColorACloud(cloud, b'cloud')
#  =============================================================================
