import numpy as np
np.random.seed(20181001)
from tensorflow import set_random_seed
set_random_seed(20181001)
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from os import listdir
import tensorflow.keras.backend as K


def rel_err(y_true, y_pred):
  return K.sum(K.abs(y_true - y_pred)) / K.sum(np.abs(y_true))

def rel_err_p(y_true, y_pred):
  return K.sum(K.abs(y_true[:, :, :, 0] - y_pred[:, :, :, 0])) / K.sum(np.abs(y_true[:, :, :, 0]))

def rel_err_vx(y_true, y_pred):
  return K.sum(K.abs(y_true[:, :, :, 1] - y_pred[:, :, :, 1])) / K.sum(np.abs(y_true[:, :, :, 1]))

def rel_err_vy(y_true, y_pred):
  return K.sum(K.abs(y_true[:, :, :, 2] - y_pred[:, :, :, 2])) / K.sum(np.abs(y_true[:, :, :, 2]))


### load data ###
#dataDir = "../tensorflow/data/trainSmallFA/"
dataDir = "../../tensorflow/data/regularFA/"
files = listdir(dataDir)[0:1000]
#files.sort()
np.random.shuffle(files)
totalLength = len(files)
inputs = np.empty((len(files), 3, 128, 128))
targets = np.empty((len(files), 3, 128, 128))

for i, file in enumerate(files):
  npfile = np.load(dataDir + file)
  d = npfile['a']
  inputs[i] = d[0:3]
  targets[i] = d[3:6]
  targets[i, 0, :, :] = (targets[i, 0, :, :] - np.mean(targets[i, 0, :, :]))
  max_x = np.amax(np.absolute(inputs[i,0,:,:]))
  max_y = np.amax(np.absolute(inputs[i,1,:,:]))
  v_2 = max_x ** 2 + max_y ** 2
  v_sqrt = np.sqrt(v_2)
  targets[i, 0, :, :] /= v_2
  targets[i, 1, :, :] /= v_sqrt
  targets[i, 2, :, :] /= v_sqrt

# normalize all samples
inputs[:,0,:,:] /= np.amax(np.absolute(inputs[:,0,:,:]))
inputs[:,1,:,:] /= np.amax(np.absolute(inputs[:,1,:,:]))
inputs[:,2,:,:] /= np.amax(np.absolute(inputs[:,2,:,:]))

targets[:,0,:,:] /= np.amax(np.absolute(targets[:,0,:,:]))
targets[:,1,:,:] /= np.amax(np.absolute(targets[:,1,:,:]))
targets[:,2,:,:] /= np.amax(np.absolute(targets[:,2,:,:]))


### define parameters ###
train_size = 0.8
validation_size = 1 - train_size

kernel_size_1 = (4, 4)
kernel_size_2 = (2, 2)

### rearrange the input data from NCHW to NHWC ###
i = int(train_size * inputs.shape[0])
j = inputs.shape[0]
#i = 0
#j = 0

inputs_c0 = inputs[:, 0]
inputs_c1 = inputs[:, 1]
inputs_c2 = inputs[:, 2]

targets_c0 = targets[:, 0]
targets_c1 = targets[:, 1]
targets_c2 = targets[:, 2]

data_input = np.ndarray((i, inputs.shape[2], inputs.shape[3], inputs.shape[1]))
data_target = np.ndarray((i, inputs.shape[2], inputs.shape[3], inputs.shape[1]))
#data_input = np.ndarray((1, inputs.shape[2], inputs.shape[3], inputs.shape[1]))
#data_target = np.ndarray((1, inputs.shape[2], inputs.shape[3], inputs.shape[1]))
data_input[:, :, :, 0] = inputs_c0[0:i]
data_input[:, :, :, 1] = inputs_c1[0:i]
data_input[:, :, :, 2] = inputs_c2[0:i]

data_target[:, :, :, 0] = targets_c0[0:i]
data_target[:, :, :, 1] = targets_c1[0:i]
data_target[:, :, :, 2] = targets_c2[0:i]

#data_input[:, :, :, 0] = inputs_c0[0]/np.max(np.abs(inputs_c0[0]))
#data_input[:, :, :, 1] = inputs_c1[0]/np.max(np.abs(inputs_c1[0]))
#data_input[:, :, :, 2] = inputs_c2[0]/np.max(np.abs(inputs_c2[0]))

#data_target[:, :, :, 0] = targets_c0[0]/np.max(np.abs(targets_c0[0]))
#data_target[:, :, :, 1] = targets_c1[0]/np.max(np.abs(targets_c1[0]))
#data_target[:, :, :, 2] = targets_c2[0]/np.max(np.abs(targets_c2[0]))

val_input = np.ndarray((j-i, inputs.shape[2], inputs.shape[3], inputs.shape[1]))
val_target = np.ndarray((j-i, inputs.shape[2], inputs.shape[3], inputs.shape[1]))
#val_input = np.ndarray((1, inputs.shape[2], inputs.shape[3], inputs.shape[1]))
#val_target = np.ndarray((1, inputs.shape[2], inputs.shape[3], inputs.shape[1]))
val_input[:, :, :, 0] = inputs_c0[i:j]
val_input[:, :, :, 1] = inputs_c1[i:j]
val_input[:, :, :, 2] = inputs_c2[i:j]

val_target[:, :, :, 0] = targets_c0[i:j]
val_target[:, :, :, 1] = targets_c1[i:j]
val_target[:, :, :, 2] = targets_c2[i:j]

#val_input[:, :, :, 0] = inputs_c0[0]/np.max(np.abs(inputs_c0[0]))
#val_input[:, :, :, 1] = inputs_c1[0]/np.max(np.abs(inputs_c1[0]))
#val_input[:, :, :, 2] = inputs_c2[0]/np.max(np.abs(inputs_c2[0]))

#val_target[:, :, :, 0] = targets_c0[0]/np.max(np.abs(targets_c0[0]))
#val_target[:, :, :, 1] = targets_c1[0]/np.max(np.abs(targets_c1[0]))
#val_target[:, :, :, 2] = targets_c2[0]/np.max(np.abs(targets_c2[0]))


# ### plot input data sample ###
# plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
# # [0] freestream field X + boundary
# plt.subplot(231)
# plt.imshow(inputs[0,0,:,:],cmap='jet')
# plt.colorbar()
# plt.title('freestream field X + boundary')

# # [1] freestream field Y + boundary
# plt.subplot(232)
# plt.imshow(inputs[0,1,:,:],cmap='jet')
# plt.colorbar()
# plt.title('freestream field Y + boundary')

# # [2] binary mask for boundary
# plt.subplot(233)
# plt.imshow(inputs[0,2,:,:],cmap='jet')
# plt.colorbar()
# plt.title('binary mask for boundary')

# # [3] pressure output
# plt.subplot(234)
# plt.imshow(targets[0,0,:,:],cmap='jet')
# plt.colorbar()
# plt.title('pressure output')

# # [4] velocity X output
# plt.subplot(235)
# plt.imshow(targets[0,1,:,:],cmap='jet')
# plt.colorbar()
# plt.title('velocity X output')

# # [5] velocity Y output
# plt.subplot(236)
# plt.imshow(targets[0,2,:,:],cmap='jet')
# plt.colorbar()
# plt.title('velocity Y output')


### create network ###
input_layer = keras.layers.Input(shape=(128, 128, 3))

factor = 2

## downsampling layers ##
convolve_0 = keras.layers.Conv2D(64//factor, kernel_size_1, strides=(1, 1), padding='same', data_format='channels_last')(input_layer)
act_0 = keras.layers.LeakyReLU(alpha=0.2)(convolve_0)

convolve_1 = keras.layers.Conv2D(64//factor, kernel_size_1, strides=(2, 2), padding='same', data_format='channels_last')(act_0)
act_1 = keras.layers.LeakyReLU(alpha=0.2)(convolve_1)

convolve_2 = keras.layers.Conv2D(128//factor, kernel_size_1, strides=(2, 2), padding='same', data_format='channels_last')(act_1)
act_2 = keras.layers.LeakyReLU(alpha=0.2)(convolve_2)

convolve_3 = keras.layers.Conv2D(256//factor//2, kernel_size_1, strides=(2, 2), padding='same', data_format='channels_last')(act_2)
act_3 = keras.layers.LeakyReLU(alpha=0.2)(convolve_3)

convolve_4 = keras.layers.Conv2D(512//factor//2, kernel_size_2, strides=(2, 2), padding='same', data_format='channels_last')(act_3)
act_4 = keras.layers.LeakyReLU(alpha=0.2)(convolve_4)

convolve_5 = keras.layers.Conv2D(512//factor//2, kernel_size_2, strides=(2, 2), padding='same', data_format='channels_last')(act_4)
act_5 = keras.layers.LeakyReLU(alpha=0.2)(convolve_5)

#convolve_6 = keras.layers.Conv2D(512//factor, kernel_size_2, strides=(2, 2), padding='same', data_format='channels_last')(act_5)

## upsampling layers with skip connections ##
#upsample_6 = keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last')(convolve_6)
#merged_6 = keras.layers.concatenate([upsample_6, act_5], axis=-1)
#deconvolve_6 = keras.layers.Conv2D(512//factor//2, kernel_size_2, strides=(1, 1), padding='same', data_format='channels_last')(merged_6)
#act_6 = keras.layers.LeakyReLU(alpha=0.0)(deconvolve_6)

upsample_5 = keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last')(act_5)
merged_5 = keras.layers.concatenate([upsample_5, act_4], axis=-1)
deconvolve_5 = keras.layers.Conv2D(256//factor//2, kernel_size_2, strides=(1, 1), padding='same', data_format='channels_last')(merged_5)
act_7 = keras.layers.LeakyReLU(alpha=0.0)(deconvolve_5)

upsample_4 = keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last')(act_7)
merged_4 = keras.layers.concatenate([upsample_4, act_3], axis=-1)
deconvolve_4 = keras.layers.Conv2D(128//factor, kernel_size_2, strides=(1, 1), padding='same', data_format='channels_last')(merged_4)
act_8 = keras.layers.LeakyReLU(alpha=0.0)(deconvolve_4)

upsample_3 = keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last')(act_8)
merged_3 = keras.layers.concatenate([upsample_3, act_2], axis=-1)
deconvolve_3 = keras.layers.Conv2D(64//factor, kernel_size_1, strides=(1, 1), padding='same', data_format='channels_last')(merged_3)
act_9 = keras.layers.LeakyReLU(alpha=0.0)(deconvolve_3)

upsample_2 = keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last')(act_9)
merged_2 = keras.layers.concatenate([upsample_2, act_1], axis=-1)
deconvolve_2 = keras.layers.Conv2D(64//factor, kernel_size_1, strides=(1, 1), padding='same', data_format='channels_last')(merged_2)
act_10 = keras.layers.LeakyReLU(alpha=0.0)(deconvolve_2)

upsample_1 = keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last')(act_10)
deconvolve_1 = keras.layers.Conv2D(64//factor, kernel_size_1, strides=(1, 1), padding='same', data_format='channels_last')(upsample_1)
act_11 = keras.layers.LeakyReLU(alpha=0.0)(deconvolve_1)

deconvolve_0 = keras.layers.Conv2D(3, kernel_size_1, strides=(1, 1), padding='same', data_format='channels_last')(act_11)


# flatten = keras.layers.Flatten(data_format='channels_last')(input_layer)

# fully = keras.layers.Dense(64*64*3)(flatten)

# reshape = keras.layers.Reshape((64, 64, 3))(fully)

#model = keras.models.Model(inputs=input_layer, outputs=reshape)

model = keras.models.Model(inputs=input_layer, outputs=deconvolve_0)


model.compile(optimizer=keras.optimizers.Adam(0.0006), loss='mean_absolute_error', metrics=[rel_err, rel_err_p, rel_err_vx, rel_err_vy])
#model.compile(optimizer=tf.train.AdamOptimizer(0.0006), loss='mean_squared_error')

print(model.summary())

### train the network ###
history = model.fit(data_input, data_target, epochs=10, batch_size=10, validation_data=(val_input, val_target), shuffle=True)
history2 = model.fit(data_input, data_target, epochs=10, batch_size=30, validation_data=(val_input, val_target), shuffle=True)
#history = model.fit(data_input, data_target, epochs=50, batch_size=1, validation_data=(data_input, data_target))
val_rel_err_data = history.history['val_rel_err'] + history2.history['val_rel_err']
rel_err_data = history.history['rel_err'] + history2.history['rel_err']
loss_data = history.history['loss'] + history2.history['loss']
val_loss_data = history.history['val_loss'] + history2.history['val_loss']
val_rel_err_p_data = history.history['val_rel_err_p'] + history2.history['val_rel_err_p']
val_rel_err_vx_data = history.history['val_rel_err_vx'] + history2.history['val_rel_err_vx']
val_rel_err_vy_data = history.history['val_rel_err_vy'] + history2.history['val_rel_err_vy']

model.save('full_network_100_samples.h5')

plt.figure()
plt.plot(loss_data)
plt.plot(val_loss_data)
plt.legend(['loss', 'val_loss'])

plt.figure()
plt.plot(rel_err_data)
plt.plot(val_rel_err_data)
plt.plot(val_rel_err_p_data)
plt.plot(val_rel_err_vx_data)
plt.plot(val_rel_err_vy_data)
plt.legend(['rel_err', 'val_rel_err', 'val_rel_err_p', 'val_rel_err_vx', 'val_rel_err_vy'])

#print(model.summary())

k=j-i
predictions = model.predict(val_input[0:k, :], batch_size=1)
truth = val_target[0:k, :]
#predictions = model.predict(val_input[:, :], batch_size=1)
#truth = val_target[:, :]

rel_error = np.sum(np.abs(data_input - data_target)) / np.sum(np.abs(data_target))
print(rel_error)

#rel_error = np.sum(np.abs(predictions - data_target)) / np.sum(np.abs(data_target))
#print(rel_error)

#predictions = np.reshape(predictions, ((-1,) + targets.shape[k:]))
#truth = np.reshape(truth, ((-1,) + targets.shape[k:]))

### plot prediction ###
plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')

# predicted data
plt.subplot(331)
plt.title('Predicted pressure')
plt.imshow(predictions[0,:,:,0], cmap='OrRd')# vmin=-100,vmax=100, cmap='jet')
plt.colorbar()
plt.subplot(332)
plt.title('Predicted x velocity')
plt.imshow(predictions[0,:,:,1], cmap='OrRd')
plt.colorbar()
plt.subplot(333)
plt.title('Predicted y velocity')
plt.imshow(predictions[0,:,:,2], cmap='OrRd')
plt.colorbar()

# ground truth data
plt.subplot(334)
plt.title('Ground truth pressure')
plt.imshow(truth[0,:,:,0],cmap='OrRd')
plt.colorbar()
plt.subplot(335)
plt.title('Ground truth x velocity')
plt.imshow(truth[0,:,:,1],cmap='OrRd')
plt.colorbar()
plt.subplot(336)
plt.title('Ground truth y velocity')
plt.imshow(truth[0,:,:,2],cmap='OrRd')
plt.colorbar()

# difference
plt.subplot(337)
plt.title('Difference pressure')
plt.imshow((truth[0,:,:,0] - predictions[0,:,:,0]),cmap='OrRd')
plt.colorbar()
plt.subplot(338)
plt.title('Difference x velocity')
plt.imshow((truth[0,:,:,1] - predictions[0,:,:,1]),cmap='OrRd')
plt.colorbar()
plt.subplot(339)
plt.title('Difference y velocity')
plt.imshow((truth[0,:,:,2] - predictions[0,:,:,2]),cmap='OrRd')
plt.colorbar()

plt.show()

