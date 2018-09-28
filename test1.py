from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, LocallyConnected2D
from keras.models import Model
from keras import backend as K
from os import listdir
import numpy as np


#import numpy as np
#import tensorflow as tf
#from tensorflow import keras
#import matplotlib.pyplot as plt
#from os import listdir
%matplotlib inline  


# load dataset
dataDir = "data/trainSmallFA/"
files = listdir(dataDir)
files.sort()

print(str(len(files)) + ' files in total')

inputs = np.zeros((len(files), 3, 64, 64))
targets = np.zeros((len(files), 3, 64, 64))

print('input dimension: ' + str(inputs.shape))

for i, file in enumerate(files):
    np_file = np.load(dataDir + file)
    d = np_file['a']
    inputs[i] = d[0:3]   # inx, iny, mask
    targets[i] = d[3:6]  # p, velx, vely


import matplotlib.pyplot as plt

def plot_sample(inputs, targets, idx):
    
    # [0] freestream field X + boundary
    plt.subplot(231)
    plt.imshow(inputs[idx, 0, :, :], cmap='jet')
    plt.colorbar()
    plt.title('freestream field X + boundary')

    # [1] freestream field Y + boundary
    plt.subplot(232)
    plt.imshow(inputs[idx, 1, :, :], cmap='jet')
    plt.colorbar()
    plt.title('freestream field Y + boundary')

    # [2] binary mask for boundary
    plt.subplot(233)
    plt.imshow(inputs[idx, 2, :, :], cmap='jet')
    plt.colorbar()
    plt.title('binary mask for boundary')

    # [3] pressure output
    plt.subplot(234)
    plt.imshow(targets[idx, 0, :, :], cmap='jet')
    plt.colorbar()
    plt.title('pressure output')

    # [4] velocity X output
    plt.subplot(235)
    plt.imshow(targets[idx, 1, :, :], cmap='jet')
    plt.colorbar()
    plt.title('velocity X output')

    # [5] velocity Y output
    plt.subplot(236)
    plt.imshow(targets[idx, 2, :, :], cmap='jet')
    plt.colorbar()
    plt.title('velocity Y output')

plot_sample(inputs, targets, 0)


#max_val = max(train_input[0,:,:])
input_vals =  [np.amax(inputs[:, i, :, :]) for i in range(inputs.shape[1])]
target_vals = [np.amax(targets[:, i, :, :]) for i in range(targets.shape[1])]

print(input_vals)
print(target_vals)

# normalize sets
for i in range(3):
    inputs[:,i,:,:] /= input_vals[i]
    targets[:,i,:,:] /= target_vals[i]

# split dataset 

TRAIN_SIZE = int(0.8*len(inputs))
train_input = inputs[:TRAIN_SIZE, :, :]
train_target = targets[:TRAIN_SIZE, :, :]

# create model
import tensorflow as tf
from keras import 

input_img = Input(shape=(3, 64, 64))
x = Conv2D(16, (3, 3), activation='relu', padding='same', data_format='channels_first')(input_img)
x = MaxPooling2D((2, 2), padding='same', data_format='channels_first')(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(x)
x = UpSampling2D((2, 2), data_format='channels_first')(x)
decoded = Conv2D(3, (3, 3), activation='linear', padding='same', data_format='channels_first')(x)

autoencoder = Model(input_img, decoded)

autoencoder.compile(optimizer=tf.train.AdamOptimizer(), loss='mean_squared_error')

print('setup finished!')

valid_input = inputs[TRAIN_SIZE:, :, :]
valid_target = inputs[TRAIN_SIZE:, :, :]

print(np.amax(inputs))

		from keras.callbacks import TensorBoard 
from keras.utils import plot_model

print(autoencoder.summary())

autoencoder.fit(train_input, train_target,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(valid_input, valid_target))
                #callbacks=[TensorBoard(log_dir='./tmp/tb', histogram_freq=0, write_graph=False)])


		