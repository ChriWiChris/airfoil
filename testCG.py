from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, LocallyConnected2D, Flatten
from keras.models import Model
from keras import backend as K
from os import listdir
import numpy as np
import matplotlib.pyplot as plt

from keras import Sequential
from keras.layers import LeakyReLU

%matplotlib inline  


MAX_NUM = 800

# load dataset
dataDir = "data/trainSmallFA/"
files = listdir(dataDir)
files.sort()

print(str(len(files)) + ' files in total')

data_len = min(MAX_NUM, len(files))

inputs = np.zeros((data_len, 3, 64, 64))
targets = np.zeros((data_len, 3, 64, 64))

for i, file in enumerate(files):
    
    if i == data_len:
        break
        
    np_file = np.load(dataDir + file)
    d = np_file['a']
    # change Data format!
    inputs[i, :] = d[:3]
    targets[i, :] = d[3:]
    
inputs = np.einsum('ijkl->iklj', inputs)
targets = np.einsum('ijkl->iklj', targets)
   
############################################################

def max_min():

    max_input_vals =  [np.amax(inputs[:, :, :, i]) for i in range(inputs.shape[-1])]
    max_target_vals = [np.amax(targets[:, :, :, i]) for i in range(targets.shape[-1])]

    min_input_vals =  [np.amin(inputs[:, :, :, i]) for i in range(inputs.shape[-1])]
    min_target_vals = [np.amin(targets[:, :, :, i]) for i in range(targets.shape[-1])]

    print(max_input_vals)
    print(max_target_vals)
    print(min_input_vals)
    print(min_target_vals)
    return max_input_vals, max_target_vals, min_input_vals, min_target_vals

[max_input_vals, max_target_vals, min_input_vals, min_target_vals] = max_min()

# normalize sets
for i in range(3):
    inputs[:,:,:,i] = 2*(inputs[:,:,:,i] - min_input_vals[i]) / (max_input_vals[i] - min_input_vals[i]) - 1
    targets[:,:,:,i] = 2*(targets[:,:,:,i] - min_target_vals[i]) / (max_target_vals[i] - min_target_vals[i]) - 1

    
# split dataset 
TRAIN_SIZE = int(0.8*len(inputs))
train_input = inputs[:TRAIN_SIZE, :, :, :]
train_target = targets[:TRAIN_SIZE, :, :, :]

valid_input = inputs[TRAIN_SIZE:, :, :, :]
valid_target = targets[TRAIN_SIZE:, :, :, :]

####################################################################

import keras

kernel_size = (4, 4)

def make_mdl():
    input_layer = keras.layers.Input(shape=(64, 64, 3))

    ## downsampling layers ##
    convolve_1 = keras.layers.Conv2D(64, kernel_size, strides=(2, 2), padding='same', activation='relu')(input_layer)
    convolve_2 = keras.layers.Conv2D(128, kernel_size, strides=(2, 2), padding='same', activation='relu')(convolve_1)
    convolve_3 = keras.layers.Conv2D(256, kernel_size, strides=(2, 2), padding='same', activation='relu')(convolve_2)
    convolve_4 = keras.layers.Conv2D(512, kernel_size, strides=(2, 2), padding='same', activation='relu')(convolve_3)
    #convolve_5 = keras.layers.Conv2D(512, kernel_size, strides=(2, 2), padding='same', activation='relu')(convolve_4)
    #convolve_6 = keras.layers.Conv2D(512, kernel_size, strides=(2, 2), padding='same', activation='relu')(convolve_5)

    ## upsampling layers with skip connections ##
    convUp1 = keras.layers.Conv2DTranspose(256, kernel_size, strides=(2, 2), padding='same', activation='elu')(convolve_4)
    merged_6 = keras.layers.concatenate([convUp1, convolve_3], axis=-1)

    convUp2 = keras.layers.Conv2DTranspose(256, kernel_size, strides=(2, 2), padding='same', activation='elu')(merged_6)
    merged_5 = keras.layers.concatenate([convUp2, convolve_2], axis=-1)

    convUp3 = keras.layers.Conv2DTranspose(128, kernel_size, strides=(2, 2), padding='same', activation='elu')(merged_5)
    merged_4 = keras.layers.concatenate([convUp3, convolve_1], axis=-1)

    upsample_1 = keras.layers.Conv2DTranspose(3, kernel_size, strides=(2, 2), padding='same', activation='tanh')(merged_4)
    
    model = keras.models.Model(inputs=input_layer, outputs=upsample_1)
    return model
		
		
###############################################################

def plot_results(predictions, truth):
    
    # make figure
    plt.figure(num=None, figsize=(20, 20), dpi=80, facecolor='w', edgecolor='k')

    # predicted data
    plt.subplot(331)
    plt.title('Predicted pressure')
    plt.imshow(predictions[0, :, :, 0], cmap='jet')  # vmin=-100,vmax=100, cmap='jet')
    plt.colorbar()
    plt.subplot(332)
    plt.title('Predicted x velocity')
    plt.imshow(predictions[0, :, :, 1], cmap='jet')
    plt.colorbar()
    plt.subplot(333)
    plt.title('Predicted y velocity')
    plt.imshow(predictions[0, :, :, 2], cmap='jet')
    plt.colorbar()

    # ground truth data
    plt.subplot(334)
    plt.title('Ground truth pressure')
    plt.imshow(truth[0, :, :, 0], cmap='jet')
    plt.colorbar()
    plt.subplot(335)
    plt.title('Ground truth x velocity')
    plt.imshow(truth[0, :, :, 1], cmap='jet')
    plt.colorbar()
    plt.subplot(336)
    plt.title('Ground truth y velocity')
    plt.imshow(truth[0, :, :, 2], cmap='jet')
    plt.colorbar()

    # difference
    plt.subplot(337)
    plt.title('Difference pressure')
    plt.imshow((truth[0, :, :, 0] - predictions[0, :, :, 0]), cmap='jet')
    plt.colorbar()
    plt.subplot(338)
    plt.title('Difference x velocity')
    plt.imshow((truth[0, :, :, 1] - predictions[0, :, :, 1]), cmap='jet')
    plt.colorbar()
    plt.subplot(339)
    plt.title('Difference y velocity')
    plt.imshow((truth[0, :, :, 2] - predictions[0,:, :, 2]), cmap='jet')
    plt.colorbar()
    plt.savefig("result.png",bbox_inches=0,pad_inches=0, dpi=15)
    #plt.show()
		
from keras.callbacks import TensorBoard 
import tensorflow as tf

NUM_EPOCH = 30

model = make_mdl()
model.compile(optimizer=tf.train.AdamOptimizer(0.002), loss='mean_absolute_error')


history = model.fit(train_input, train_target,
                epochs=NUM_EPOCH,
                batch_size=32,
                shuffle=True,
                validation_data=(valid_input, valid_target))

print(model.summary())

#predictions = model.predict(valid_input[0:1, :, :, :])
#truth = valid_target[0:1, :, :, :]

predictions = model.predict(train_input[0:1, :, :, :])
truth = train_target[0:1, :, :, :]


print(predictions.shape)
print(truth.shape)

plot_results(predictions, truth)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(history.history['loss'])
ax = fig.add_subplot(122)
ax.plot(history.history['val_loss'], color='red')



print('dimension: ' + str(inputs.shape))
print('done')  



