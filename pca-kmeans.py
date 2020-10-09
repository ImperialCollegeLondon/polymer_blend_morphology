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

# Perform clustering and append the datalabels to the dataframe
kmeanModel = KMeans(n_clusters=6,init='k-means++').fit(X)
dimred_df["Cluster labels"] = kmeanModel.labels_
print("Finished clustering")

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='small')
fig, ax = plt.subplots(figsize=(8,4.5))

for i in range(0,len(dimred_df["Cluster labels"])):
    if dimred_df["Cluster labels"][i] == 0:
        c1 = ax.scatter(dimred_df[0][i], dimred_df[1][i], c="r")
    elif dimred_df["Cluster labels"][i] == 1:
        c2 = ax.scatter(dimred_df[0][i], dimred_df[1][i], c="g")
    elif dimred_df["Cluster labels"][i] == 2:
        c3 = ax.scatter(dimred_df[0][i], dimred_df[1][i], c="b")
    elif dimred_df["Cluster labels"][i] == 3:
        c4 = ax.scatter(dimred_df[0][i], dimred_df[1][i], c="c")
    elif dimred_df["Cluster labels"][i] == 4:
        c5 = ax.scatter(dimred_df[0][i], dimred_df[1][i], c="m")
    elif dimred_df["Cluster labels"][i] == 5:
        c6 = ax.scatter(dimred_df[0][i], dimred_df[1][i], c="k")

# plt.scatter(dimred_df[0], dimred_df[1], c=dimred_df["Cluster labels"], cmap="rainbow", s=30)
ax.legend([c1,c2,c3,c4,c5,c6],["0", "1", "2", "3", "4", "5"])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()