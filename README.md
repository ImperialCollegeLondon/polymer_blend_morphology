# polymer_blend_morphology
Applying ML to study simulated polymer blend morphology

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



