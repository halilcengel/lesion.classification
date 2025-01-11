import matplotlib.pyplot as plt
import numpy as np
import skimage.io
from scipy import ndimage as ndi
from sklearn.cluster import DBSCAN
img = skimage.io.imread('images/rapor/hair_removal/ISIC_0000264.jpg')[:,:,0]


image_max = ndi.maximum_filter(-img, size=10, mode='constant')
image_max = image_max > np.quantile(image_max, 0.8)

X = np.array(np.nonzero(image_max)).transpose()
clustering = DBSCAN(eps=10, min_samples=200).fit(X)

fig, axs = plt.subplots(ncols = 2, sharex = True, sharey = True)
axs[0].imshow(image_max, cmap=plt.cm.gray)
clustering.labels_[clustering.labels_ == -1] = max(clustering.labels_)+1
axs[1].scatter(X[:,1], X[:,0], c = clustering.labels_)
plt.show()