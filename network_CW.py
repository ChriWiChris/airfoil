import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from os import listdir

### load data ###
dataDir = "../tensorflow/data/trainSmallFA/"
files = listdir(dataDir)
files.sort()
totalLength = len(files)
inputs = np.empty((len(files), 3, 64, 64))
targets = np.empty((len(files), 3, 64, 64))

for i, file in enumerate(files):
  npfile = np.load(dataDir + file)
  d = npfile['a']
  inputs[i] = d[0:3]
  targets[i] = d[3:6]

### plot input data sample ###
plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
# [0] freestream field X + boundary
plt.subplot(231)
plt.imshow(inputs[0,0,:,:],cmap='jet')
plt.colorbar()
plt.title('freestream field X + boundary')

# [1] freestream field Y + boundary
plt.subplot(232)
plt.imshow(inputs[0,1,:,:],cmap='jet')
plt.colorbar()
plt.title('freestream field Y + boundary')

# [2] binary mask for boundary
plt.subplot(233)
plt.imshow(inputs[0,2,:,:],cmap='jet')
plt.colorbar()
plt.title('binary mask for boundary')

# [3] pressure output
plt.subplot(234)
plt.imshow(targets[0,0,:,:],cmap='jet')
plt.colorbar()
plt.title('pressure output')

# [4] velocity X output
plt.subplot(235)
plt.imshow(targets[0,1,:,:],cmap='jet')
plt.colorbar()
plt.title('velocity X output')

# [5] velocity Y output
plt.subplot(236)
plt.imshow(targets[0,2,:,:],cmap='jet')
plt.colorbar()
plt.title('velocity Y output')

### define parameters ###
train_size = 0.8
validation_size = 1 - train_size

kernel_size = (3, 3)

### create network ###
input_layer = keras.layers.Input(shape=(64, 64, 3))

## downsampling layers ##
convolve_1 = keras.layers.Conv2D(64, kernel_size, strides=(2, 2), padding='same', data_format='channels_last')(input_layer)
convolve_2 = keras.layers.Conv2D(128, kernel_size, strides=(2, 2), padding='same', data_format='channels_last')(convolve_1)
convolve_3 = keras.layers.Conv2D(256, kernel_size, strides=(2, 2), padding='same', data_format='channels_last')(convolve_2)
convolve_4 = keras.layers.Conv2D(512, kernel_size, strides=(2, 2), padding='same', data_format='channels_last')(convolve_3)
convolve_5 = keras.layers.Conv2D(512, kernel_size, strides=(2, 2), padding='same', data_format='channels_last')(convolve_4)
convolve_6 = keras.layers.Conv2D(512, kernel_size, strides=(2, 2), padding='same', data_format='channels_last')(convolve_5)

## upsampling layers with skip connections ##
upsample_6 = keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last')(convolve_6)
deconvolve_6 = keras.layers.Conv2D(512, kernel_size, strides=(1, 1), padding='same', data_format='channels_last')(upsample_6)
merged_6 = keras.layers.concatenate([deconvolve_6, convolve_5], axis=-1)

upsample_5 = keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last')(merged_6)
deconvolve_5 = keras.layers.Conv2D(512, kernel_size, strides=(1, 1), padding='same', data_format='channels_last')(upsample_5)
merged_5 = keras.layers.concatenate([deconvolve_5, convolve_4], axis=-1)

upsample_4 = keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last')(merged_5)
deconvolve_4 = keras.layers.Conv2D(256, kernel_size, strides=(1, 1), padding='same', data_format='channels_last')(upsample_4)
merged_4 = keras.layers.concatenate([deconvolve_4, convolve_3], axis=-1)

upsample_3 = keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last')(merged_4)
deconvolve_3 = keras.layers.Conv2D(128, kernel_size, strides=(1, 1), padding='same', data_format='channels_last')(upsample_3)
merged_3 = keras.layers.concatenate([deconvolve_3, convolve_2], axis=-1)

upsample_2 = keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last')(merged_3)
deconvolve_2 = keras.layers.Conv2D(64, kernel_size, strides=(1, 1), padding='same', data_format='channels_last')(upsample_2)
merged_2 = keras.layers.concatenate([deconvolve_2, convolve_1], axis=-1)

upsample_1 = keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last')(merged_2)
deconvolve_1 = keras.layers.Conv2D(3, kernel_size, strides=(1, 1), padding='same', data_format='channels_last')(upsample_1)

model = keras.models.Model(inputs=input_layer, outputs=deconvolve_1)

model.compile(optimizer=tf.train.AdamOptimizer(0.0001), loss='mean_squared_error', metrics=['accuracy'])


### rearrange the input data from NCHW to NHWC ###
i = int(train_size * inputs.shape[0])
j = inputs.shape[0]
# data_input = np.reshape(inputs[0:i], (i,-1))
# data_target = np.reshape(targets[0:i], (i, -1))

inputs_c0 = inputs[:, 0]
inputs_c1 = inputs[:, 1]
inputs_c2 = inputs[:, 2]

targets_c0 = targets[:, 0]
targets_c1 = targets[:, 1]
targets_c2 = targets[:, 2]

data_input = np.ndarray((i, inputs.shape[2], inputs.shape[3], inputs.shape[1]))
data_target = np.ndarray((i, inputs.shape[2], inputs.shape[3], inputs.shape[1]))
data_input[:, :, :, 0] = inputs_c0[0:i]
data_input[:, :, :, 1] = inputs_c1[0:i]
data_input[:, :, :, 2] = inputs_c2[0:i]

data_target[:, :, :, 0] = targets_c0[0:i]
data_target[:, :, :, 1] = targets_c1[0:i]
data_target[:, :, :, 2] = targets_c2[0:i]

val_input = np.ndarray((j-i, inputs.shape[2], inputs.shape[3], inputs.shape[1]))
val_target = np.ndarray((j-i, inputs.shape[2], inputs.shape[3], inputs.shape[1]))
val_input[:, :, :, 0] = inputs_c0[i:j]
val_input[:, :, :, 1] = inputs_c1[i:j]
val_input[:, :, :, 2] = inputs_c2[i:j]

val_target[:, :, :, 0] = targets_c0[i:j]
val_target[:, :, :, 1] = targets_c1[i:j]
val_target[:, :, :, 2] = targets_c2[i:j]


### train the network ###
history = model.fit(data_input, data_target, epochs=1, batch_size=10, validation_data=(val_input, val_target))

print(model.summary())

k=j-i
predictions = model.predict(val_input[0:k, :], batch_size=1)
truth = val_target[0:k, :]

#predictions = np.reshape(predictions, ((-1,) + targets.shape[k:]))
#truth = np.reshape(truth, ((-1,) + targets.shape[k:]))

### plot prediction ###
plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')

# predicted data
plt.subplot(331)
plt.title('Predicted pressure')
plt.imshow(predictions[0,:,:,0], cmap='jet')# vmin=-100,vmax=100, cmap='jet')
plt.colorbar()
plt.subplot(332)
plt.title('Predicted x velocity')
plt.imshow(predictions[0,:,:,1], cmap='jet')
plt.colorbar()
plt.subplot(333)
plt.title('Predicted y velocity')
plt.imshow(predictions[0,:,:,2], cmap='jet')
plt.colorbar()

# ground truth data
plt.subplot(334)
plt.title('Ground truth pressure')
plt.imshow(truth[0,:,:,0],cmap='jet')
plt.colorbar()
plt.subplot(335)
plt.title('Ground truth x velocity')
plt.imshow(truth[0,:,:,1],cmap='jet')
plt.colorbar()
plt.subplot(336)
plt.title('Ground truth y velocity')
plt.imshow(truth[0,:,:,2],cmap='jet')
plt.colorbar()

# difference
plt.subplot(337)
plt.title('Difference pressure')
plt.imshow((truth[0,:,:,0] - predictions[0,:,:,0]),cmap='jet')
plt.colorbar()
plt.subplot(338)
plt.title('Difference x velocity')
plt.imshow((truth[0,:,:,1] - predictions[0,:,:,1]),cmap='jet')
plt.colorbar()
plt.subplot(339)
plt.title('Difference y velocity')
plt.imshow((truth[0,:,:,2] - predictions[0,:,:,2]),cmap='jet')
plt.colorbar()

plt.show()

