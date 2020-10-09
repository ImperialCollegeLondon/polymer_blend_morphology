# Imports
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D

import numpy as np 
import pandas as pd 
import glob as glob

## Machine learning imports
from sklearn.decomposition import PCA
from skimage import data, io, color

# Import k-means stuff
from sklearn.cluster import KMeans

image_store = "./Images"


# This section reads in the image data
name_list = []
channel_0 = Path(image_store).glob("image_*_0.png")
for image_path in channel_0:
    stem = re.search(r"(image_\d+)", image_path.as_posix())
    name = stem.group(1)
    name_list.append(name)
category = ["unknown"] * len(name_list)

all_flat = []
for i, name in enumerate(name_list):
    img_flat = np.empty(0)
    for channel in [0, 1, 2]:
        img = io.imread(f"{image_store}/{name}_{channel}.png", as_gray=True)
        img_flat = np.append(img_flat, img.flatten())
    all_flat.append(img_flat)

df = pd.DataFrame(all_flat)
df.index = name_list

print("Finished loading images")

# Run PCA
# Specify number of dimensions retained
dim = 2
dimred = PCA(n_components=dim).fit_transform(df)
dimred_df = pd.DataFrame(dimred)
dimred_df["label"] = name_list
dimred_df["cat"] = category

print("Finished running PCA")


# Run Elbow 
df_heading_list = []
for i in range(dim):
    df_heading_list.append(i)

# Extract PC data
X = np.array(dimred_df[df_heading_list])

distortions = []
K = range(1,31)
for k in K:
    kmeanModel = KMeans(n_clusters=k,init='k-means++').fit(X)
    kmeanModel.fit(X)
    distortions.append(kmeanModel.inertia_)
#    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1))/ X.shape[0])

wcss = distortions
l_dist = []
for i in K:
    p1 = np.array([K[0],wcss[0]])
    p2 = np.array([K[-1],wcss[-1]])
    p = np.array([i,wcss[i-1]])
    l_dist.append(np.linalg.norm(np.cross(p2-p1,p1-p))/np.linalg.norm(p2-p1))

# Plot the elbow
fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.plot(K, distortions, color=color,marker='x')
ax1.set_xlabel('k')
ax1.set_ylabel('WCSS',color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2=ax1.twinx() #initiate second axes with same x axis
color ='tab:red'
ax2.plot(K,l_dist,color=color,marker='o')
ax2.set_ylabel('Distance',color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('The Elbow Method showing the optimal k')
plt.show()



