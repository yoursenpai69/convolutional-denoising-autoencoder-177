# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
Denoising autoencoders create a corrupted copy of the input by introducing some noise. 
This helps to avoid the autoencoders to copy the input to the output without learning features about the data. 
These autoencoders take a partially corrupted input while training to recover the original undistorted input. 

The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.

![image](https://user-images.githubusercontent.com/57000479/201095595-646bab7c-6f00-472a-b9ba-7d2d8d584b10.png)

## Convolution Autoencoder Network Model

![image](https://user-images.githubusercontent.com/57000479/201090797-660dafee-646a-4d00-87a9-81965a5264da.png)

## DESIGN STEPS

### STEP 1:

Download and load the dataset to colab.

### STEP 2:

Split the data into train and test.

### STEP 3:

Introduce Noise into the dataset

### STEP 4:

Build the Neural Network

### STEP 5:

Train the model with training data

### STEP 6:

Evaluate the model with the testing data

### STEP 7:

View the denoised output

## PROGRAM

```python
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
(x_train, _), (x_test, _) = mnist.load_data()
x_train.shape
x_train_scaled = x_train.astype('float32') / 255.
x_test_scaled = x_test.astype('float32') / 255.
x_train_scaled = np.reshape(x_train_scaled, (len(x_train_scaled), 28, 28, 1))
x_test_scaled = np.reshape(x_test_scaled, (len(x_test_scaled), 28, 28, 1))
noise_factor = 0.5
x_train_noisy = x_train_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train_scaled.shape) 
x_test_noisy = x_test_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test_scaled.shape) 

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

n = 10
plt.figure(figsize=(20, 2))
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

input_img = keras.Input(shape=(28, 28, 1))

# Write your encoder here
l = layers.Conv2D(32,(3,3),activation = 'relu',padding='same')(input_img)
l = layers.MaxPooling2D((2, 2), padding='same')(l)
l = layers.Conv2D(32,(3,3),activation = 'relu',padding='same')(l)
l = layers.MaxPooling2D((2, 2), padding='same')(l)
l = layers.Conv2D(32,(3,3),activation = 'relu',padding='same')(l)
encoded = layers.MaxPooling2D((2, 2), padding='same')(l)

# Encoder output dimension is ## (7,7,32) ##
# Write your decoder here
l = layers.Conv2D(32,(3,3),activation = 'relu',padding='same')(encoded)
l = layers.UpSampling2D((2,2))(l)
l = layers.Conv2D(32,(3,3),activation = 'relu',padding='same')(l)
l = layers.UpSampling2D((2,2))(l)
l = layers.Conv2D(32,(3,3),activation = 'relu')(l)
l = layers.UpSampling2D((2,2))(l)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(l)

autoencoder = keras.Model(input_img, decoded)
autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train_noisy, x_train_scaled,
                epochs=2,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test_scaled))
decoded_imgs = autoencoder.predict(x_test_noisy)        
n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(3, n, i)
    plt.imshow(x_test_scaled[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display noisy
    ax = plt.subplot(3, n, i+n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)    

    # Display reconstruction
    ax = plt.subplot(3, n, i + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```

## OUTPUT

### Original vs Noisy Vs Reconstructed Image

![image](https://user-images.githubusercontent.com/57000479/201095302-b0ed0ebd-9ef1-480e-8977-5cbb49bfae49.png)


## RESULT
Successfully developed a convolutional autoencoder for image denoising application.
