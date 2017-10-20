'''
The base of this code is layed by https://github.com/Zackory/Keras-MNIST-GAN/blob/master/mnist_gan.py and was written by Zackory Erickson
'''

import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from keras.layers import Input, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D
from keras.models import Model, Sequential, load_model
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.datasets import cifar10
from keras.optimizers import Adam

from keras.regularizers import l1_l2

#------
# DATA
#------
from keras import backend as K
K.set_image_dim_ordering('th')

import h5py

# Get hdf5 file
hdf5_file = os.path.join("PATH TO DATASET", "CelebA_Transfer_64_data.h5")

with h5py.File(hdf5_file, "r") as hf:
    X_train = hf["data"] [()] #[()] makes it read the whole thing    
    #X_train = X_train[:50000]  
    X_train = X_train.astype(np.float32) / 255        
      

#----------------
# HYPERPARAMETERS
#----------------
randomDim = 100

adam = Adam(lr=0.0002, beta_1=0.5)

reg = lambda: l1_l2(l1=1e-7, l2=1e-7)

dropout = 0

# Load the old models
old_discriminator = 'dcgan_discriminator_epoch_19.h5'
old_generator = 'dcgan_discriminator_epoch_19.h5'

# There is a strange bug if the optimizer is loaded from last network therefore just delete them
with h5py.File(old_generator, 'a') as f:
    if 'optimizer_weights' in f.keys():
            del f['optimizer_weights'] 

with h5py.File(old_discriminator, 'a') as f:
    if 'optimizer_weights' in f.keys():
            del f['optimizer_weights']
         

generator = load_model(old_generator)
discriminator = load_model(old_discriminator)



#-----
# GAN
#-----
discriminator.trainable = False
ganInput = Input(shape=(randomDim,))
x = generator(ganInput)
ganOutput = discriminator(x)
gan = Model(inputs=ganInput, outputs=ganOutput)
gan.compile(loss='binary_crossentropy', optimizer=adam)

#-----------
# FUNCTIONS
#-----------
dLosses = []
gLosses = []
def plotLoss(epoch):
    assertExists('images')

    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminative loss')
    plt.plot(gLosses, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()    
    plt.savefig('images/dcgan_loss_epoch_%d.png' % epoch)

# Create a wall of generated images
examples=100
noise = np.random.normal(0, 1, size=[examples, randomDim])
def plotGeneratedImages(epoch, dim=(10, 10), figsize=(10, 10)):
    generatedImages = generator.predict(noise)
    generatedImages = generatedImages.transpose(0, 2, 3, 1)

    assertExists('images')

    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generatedImages[i, :, :, :], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()    
    plt.savefig('images/dcgan_generated_image_epoch_%d.png' % epoch)


# Save the generator and discriminator networks (and weights) for later use
def savemodels(epoch):
    assertExists('models')
    generator.save('models/transfer_dcgan_generator_epoch_%d.h5' % epoch)
    discriminator.save('models/transfer_dcgan_discriminator_epoch_%d.h5' % epoch)


def train(epochs=1, batchSize=128, save_interval=1, start_at=1):
    batchCount = X_train.shape[0] // batchSize
    print('Epochs:', epochs)
    print('Batch size:', batchSize)
    print('Batches per epoch:', batchCount)

    #show what the images looked like before
    plotGeneratedImages(0)

    for e in range(start_at, epochs+1):
        print('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in tqdm(range(batchCount)):
            # Get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            imageBatch = X_train[np.random.randint(0, X_train.shape[0], size=batchSize)]

            # Generate fake images
            generatedImages = generator.predict(noise)

            X = np.concatenate([imageBatch, generatedImages])

            # Labels for generated and real data
            yDis = np.zeros(2*batchSize)
            # One-sided label smoothing = not exactly 1
            yDis[:batchSize] = 0.95

            # Train discriminator
            discriminator.trainable = True
            dloss = discriminator.train_on_batch(X, yDis)

            # Train generator
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            yGen = np.ones(batchSize)
            discriminator.trainable = False
            gloss = gan.train_on_batch(noise, yGen) #<-- only generator is trained because discr. is not trainable
            #EXPLANATION: generator tries to make the discriminator output 1!

        # Store loss of most recent batch from this epoch
        dLosses.append(dloss)
        gLosses.append(gloss)

        #plot after every epoch
        if (e == 1 or e % save_interval == 0):
            plotGeneratedImages(e)
            savemodels(e)

    # Plot losses from every epoch
    plotLoss(e)


def assertExists(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':	
    train(100, 128, 1) #test2: 100

