# polymer_blend_morphology
Applying ML to study simulated ternary polymer blend morphology using a modified Cahn-Hilliard framework. The preprint of the accepted paper can be found here: https://arxiv.org/abs/2007.07276. 

This resposity contains sample code to do the following: 1) Perform the Cahn-Hilliard simulations using Fenics 2019 with a Singularity container from Ocellaris. 2) Perform the various dimensionality reduction and clustering techniques on the dataset and 3) Run the Gaussian process classifier to generate the morphology maps. 

## Running the Cahn-Hilliard simulation
To create the appropriate Singularity container, run the following which generates a Singularity with the necessary dependencies:
```
singularity pull library://trlandet/default/ocellaris:2019.0.2
```

An alternative for setting up a suitable environment is to use an Anaconda environment with Fenics 2019.1.0. More information can be found at https://fenicsproject.org/download/. 

With Singularity, run the demo code `ternary.py` as follows (assuming the container is in the current directory)
```
singularity exec ./ocellaris_2019.0.2.sif python3 ternary.py
```

The post-processing can be carried out using ParaView (https://www.paraview.org/). 


## Dataset
The images generated during the simulations carried out as part of the study are found in the `Images` directory. The filename e.g. `image_0002.png` captures the run ID (`0002`). The input parameters and cluster labels for that specific run ID can be found in `Clustering Results.csv`. 

The dataset also contains the extracted Red, Green, Blue (RGB) channels of the original image. `image_*_0` corresponds to Blues, `image_*_1` corresponds to Greens and `image_*_2` corresponds to Reds. 

## Performing PCA & K-Means clustering
The next three sections will require scikit-learn and/or Tensorflow (using Keras). A YAML environment file, `image.yml`, of the Anaconda environment used is provided for the interested user. 

The first file is `elbow.py` which can be used to estimate the optimal number of clusters for a given number of PCs retained using K-Means clustering. The code in `elbow.py` can be modified to generate the elbow curve plots for multiple conditions so that the user can identify the optimal number of clusters for a range of parameters e.g. the number of PCs retained. 

The second file is `pca-kmeans.py` which is used to generate 2/3D scatter plot of the clusters given a specification of the optimal number of clusters determined by the elbow method. There is a certain amount of variability associated with the identification of the number of clusters as there is no distinct "Elbow perse" in many of the plots. This result is evidenced in Figure 4 of the manuscript. 

These two scripts can be easily modified to implement other dimensionality reduction techniques within scikit-learn (https://scikit-learn.org/stable/modules/manifold.html#manifold). However, the choice of a clustering algorithm requires careful thought as outlined in the manuscript. A good reference on the various clustering algorithms can be found here :https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py. 

## Performing dimensionality reduction using a convolutional autoencoder
The script `convae.py` contains two examples of convolutional autoencoder architectures that can be used to encode the entire dataset of images. Details of the various parameters used and the corresponding results are available in the manuscript. Densely connected autoencoders or other autoencoder architectures can be implemented comparatively easily (https://blog.keras.io/building-autoencoders-in-keras.html). But as noted in the manuscript, the memory requirements for Dense autoencoders can be quite high despite yielding mediocre results. 

The script outputs a .csv file which contains the encoded data. At this point, one could perform further dimensionality reduction using the scripts above (with some modifications to work with CSV files) or perform clustering on the encoded data directly. 

## Performing Gaussian process classification for prediction
The script `gpc.py` can be used to run Gausian process classification for generating the regime maps as shown in figure 10 of the manuscript. A helper function in the script `extract_slices.py` is necessary to extract the appropriate slice of the full dataset for performing prediction. Additional detail regarding the types of slices that can be generated can be found in the comments of `extract_slices.py`. Currently, it is configured to run using the `Clustering_Results.csv` file in the repo as the dataset which contains the results clustering results from the various clustering techniques. We found manual clustering to be the most effective. 


