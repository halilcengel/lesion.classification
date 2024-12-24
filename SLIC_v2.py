import matplotlib.pyplot as plt
from skimage import io, segmentation, color
#USE

img_path = "segmentation_v2_masked_images/ISIC_0000042_masked.png"

image = io.imread(img_path)

segments_slic = segmentation.slic(image, n_segments=32, compactness=10, sigma=1)

segmented_image = color.label2rgb(segments_slic, image, kind='avg')

fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
ax[0].imshow(image)
ax[0].set_title('Original Image')
ax[1].imshow(segmented_image)
ax[1].set_title('SLIC Segmented Image')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()