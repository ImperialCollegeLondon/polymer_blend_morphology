from keras.preprocessing.image import img_to_array
from keras import regularizers
from keras.layers import Dense, Input, Flatten, Reshape, InputLayer, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Cropping2D
from keras.models import Model, Sequential

import pandas as pd
import glob as glob
import re 
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Image location
relative_image_folder = "./Images/"

# Create a dataframe of image arrays
print ("[INFO] loading images...")
allimagepaths = glob.glob(relative_image_folder + "image_*.png")
channelsimagepaths = glob.glob(relative_image_folder + "image_*_*.png")

# We need to subtract the path of the specific channels to just get the main images
imagepaths = list (set(allimagepaths) - set(channelsimagepaths))
#Sort the list so the numbers are nicely aligned
imagepaths.sort()

# Initialise lists for the labels and data
labels = []
data = []
for imagepathstr in imagepaths:
    # Use the keras preprocessing tool
    image = cv2.imread(imagepathstr)
    # This function seems to create the image as a numpy array
    image = img_to_array(image)
    data.append(image)

    # Extract the label from that part of the string
    m = re.search("image_(.+?).png", imagepathstr)
    label = m.group(1)
    labels.append(label)
print ("[INFO] Images have been loaded, preparing data I/O...")

"""
Prepping the data. This includes normalisation
"""
# Prepare Input
X = np.array(data, dtype="float") / 255.0 # Normalise the data
labels = np.array(labels)


# Setting up the autoencoder
# Gives the number of pixels on one axis 
input_image_shape = X.shape[1:]
print (input_image_shape)

print("[INFO] Done normalising, preparing the autoencoder now... ")


# Example of a Conv3 autoencoder
def conv_3l_dense_0l_autoencoder (img_shape):
    # This current model is intractable
    encoder = Sequential()
    encoder.add(InputLayer(img_shape))
    encoder.add(Conv2D(32, (3,3), activation="relu", padding="same"))
    encoder.add(MaxPooling2D((2), padding="same"))
    encoder.add(Conv2D(32, (3,3), activation="relu", padding="same"))
    encoder.add(MaxPooling2D((2), padding="same"))
    encoder.add(Conv2D(1, (3,3), activation="relu", padding="same"))
    encoder.add(MaxPooling2D((2), padding="same"))
    encoder.add(Flatten())

    # The decoder
    decoder = Sequential()
    decoder.add(InputLayer((25*25*1,)))
    decoder.add(Reshape((25,25,1)))
    decoder.add(Conv2D(1, (3,3), activation="relu", padding="same"))
    decoder.add(UpSampling2D(2))
    decoder.add(Conv2D(32, (3,3), activation="relu", padding="same"))
    decoder.add(UpSampling2D(2))
    decoder.add(Conv2D(32, (3,3), activation="relu", padding="same"))
    decoder.add(UpSampling2D(2))
    decoder.add(Conv2D(3, (3,3), activation="sigmoid", padding="same"))

    return encoder, decoder

# Example of a Conv4 autoencoder
def conv_4l_dense_0l_autoencoder (img_shape):
    # This current model is intractable
    encoder = Sequential()
    encoder.add(InputLayer(img_shape))
    encoder.add(Conv2D(32, (3,3), activation="relu", padding="same"))
    encoder.add(MaxPooling2D((2), padding="same"))
    encoder.add(Conv2D(32, (3,3), activation="relu", padding="same"))
    encoder.add(MaxPooling2D((2), padding="same"))
    encoder.add(Conv2D(32, (3,3), activation="relu", padding="same"))
    encoder.add(MaxPooling2D((2), padding="same"))
    encoder.add(Conv2D(4, (3,3), activation="relu", padding="same"))
    encoder.add(MaxPooling2D((2), padding="same"))
    encoder.add(Flatten())

    # The decoder
    decoder = Sequential()
    decoder.add(InputLayer((13*13*4,)))
    decoder.add(Reshape((13,13,4)))
    decoder.add(Conv2D(4, (3,3), activation="relu", padding="same"))
    decoder.add(UpSampling2D(2))
    decoder.add(Conv2D(32, (3,3), activation="relu", padding="same"))
    decoder.add(UpSampling2D(2))
    decoder.add(Conv2D(32, (3,3), activation="relu", padding="same"))
    decoder.add(UpSampling2D(2))
    decoder.add(Conv2D(32, (3,3), activation="relu", padding="same"))
    decoder.add(UpSampling2D(2))
    decoder.add(Conv2D(3, (3,3), activation="sigmoid", padding="same"))
    decoder.add(Cropping2D(cropping=((4,4), (4,4)), input_shape=(208, 208,3)))

    return encoder, decoder


encoder,decoder = conv_4l_dense_0l_autoencoder(input_image_shape)
inp = Input(input_image_shape)
code = encoder(inp)
reconstruction = decoder(code)
autoencoder = Model(inp, reconstruction)
autoencoder.compile(optimizer="adam", loss="logcosh", metrics=["acc"])
print(autoencoder.summary())

history = autoencoder.fit(X, X,
                        epochs=1,
                        batch_size=50,
                        shuffle = True,
                        validation_split = 0.0,
                        validation_data=None
)

encoded_imgs = encoder.predict(X)
decoded_imgs = decoder.predict(encoded_imgs)

df = pd.DataFrame(encoded_imgs)
df["label"] = labels
df.to_csv("./4conv_0dense_bottleneck_676dim_shuffletrue.csv")
print("[INFO] Done exporting csv of bottleneck values...")