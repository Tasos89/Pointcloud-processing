#%%
#pca 1
# 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn import decomposition, datasets
from sklearn.preprocessing import StandardScaler

data = np.concatenate((xyz_new),axis=0)
scaler = MinMaxScaler(feature_range=[0,1])
data_rescaled = scaler.fit_transform(xyz_new)
pca = PCA().fit(data_rescaled)

plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)')
plt.title('Pulsar Dataset Explained Variance')
plt.show()

#%%
#pca2
# https://chrisalbon.com/machine_learning/feature_engineering/feature_extraction_with_pca/

import numpy as np
from sklearn import decomposition, datasets
from sklearn.preprocessing import StandardScaler

features = np.asarray(features)
sc = StandardScaler()
#fit the scaler to the features and transform
features_std = sc.fit_transform(features)
#create a pca object with the 8 components as parameter
pca = decomposition.PCA(n_components=5)
#fit the PCA and transform the data
features_std_pca = pca.fit_transform(features_std)
features_std_pca.shape


