# # Some neccesary libraries and functions

# +
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Comment this line for using the GPU

from tensorflow.keras.layers import Input, Dense, Dropout, AveragePooling2D, Flatten, Conv2D
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import h5py


# -

def MakePlots(model, x_testset, y_testset, modelName,irun):
    real_masses = np.exp(y_testset) # If exp
    pred_masses = np.exp(model.predict(x_testset))
    diff = real_masses - pred_masses
    
    log_real_masses = np.log(real_masses)
    log_pred_masses= np.log(pred_masses)
    diff_log = log_real_masses - log_pred_masses
    
    mse = np.mean(diff**2, 0)
    std = np.std(diff**2, 0)
    mse_rel = np.mean((diff / real_masses) ** 2, 0)
    std_rel = np.std((diff / real_masses) ** 2, 0)
    mae = np.mean(np.abs(diff), 0)
    mae_std = np.std(np.abs(diff), 0)
    mae_rel = np.mean(np.abs(diff / real_masses), 0)
    std_rel = np.std(np.abs(diff / real_masses), 0)
        
    mse_log = np.mean(diff_log**2, 0)
    std_log = np.std(diff_log**2, 0)
    mse_log_rel = np.mean((diff_log / log_real_masses) ** 2, 0)
    std_log_rel = np.std((diff_log / log_real_masses) ** 2, 0)
    mae_log = np.mean(np.abs(diff_log), 0)
    mae_log_std = np.std(np.abs(diff_log), 0)
    mae_log_rel = np.mean(np.abs(diff_log / log_real_masses), 0)
    std_log_rel = np.std(np.abs(diff_log / log_real_masses), 0)
    
    # Residuals
    plt.cla()
    plt.clf()
    plt.xlabel(r'$\Delta \log M$')
    plt.ylabel(r'P($\Delta \log M$)')
    for i in range(20):
        plt.hist(diff_log[:,i], bins=20, density=True, histtype='step', alpha=1, label=r'$r = %s \; \mathrm{kpc}$'%np.around(bins[i], 1))
    #plt.legend(loc='upper left', fontsize='small')
    plt.grid(color='grey', linestyle=':', linewidth=0.25)
    plt.tight_layout()
    plt.savefig('../data/models/' + modelName + 'graph/residuals_' + str(irun) +'.pdf')
    
    # MSE and MAE
    plt.cla()
    plt.clf()
    plt.xlabel(r'$r$ [kpc]')
    plt.ylabel(r'$\Delta$')
    plt.plot(np.around(bins, 1), mse_rel, label=r'$\left< (\frac{\Delta M}{M})^{2} \right>(r)$')
    plt.plot(np.around(bins, 1), mae_rel, label=r'$\left< \left|\Delta M / M \right| \right>(r)$')
    plt.grid(color='grey', linestyle=':', linewidth=0.25, which='both')
    plt.xscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../data/models/' + modelName + 'graph/MSE_MAE_' + str(irun) + '.pdf')
    
    # MSE and MAE in log
    plt.cla()
    plt.clf()
    plt.xlabel(r'$r$ [kpc]')
    plt.ylabel(r'$\Delta$')
    plt.plot(np.around(bins, 1), mse_log_rel, label=r'$\left< (\frac{\Delta \mu}{\mu})^{2} \right>(r)$')
    plt.plot(np.around(bins, 1), mae_log_rel, label=r'$\left< \left|\Delta \mu / \mu \right| \right>(r)$')
    plt.grid(color='grey', linestyle=':', linewidth=0.25, which='both')
    plt.xscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../data/models/' + modelName + 'graph/MSE_MAE_log_' + str(irun) + '.pdf')
    
    # Scatter-plots
    plt.cla()
    plt.clf()
    fig, axes          = plt.subplots(5, figsize=(5, 50), sharex=True)
    mass_min, mass_max = int(np.amin([log_real_masses, log_pred_masses])), int(np.amax([log_real_masses, log_pred_masses])) + 1
    ticks              = np.arange(mass_min, mass_max + 1)

    j = 0
    for i in range(20):
        if (i%5 == 0):
            ax = axes[j,]
            ax.plot((mass_min, mass_max), (mass_min, mass_max), color='black', ls='--', lw=1, zorder=2)
            ax.scatter(log_real_masses[:,i], log_pred_masses[:,i], s=0.001, zorder=3)
            ax.set_aspect('equal', 'box')
            ax.set_title(r'log $M_\mathrm{DM}(r \leq %s \mathrm{kpc})$' % np.around(bins[i], 1))
            ax.text(0.05, 0.95, r'$\left< \left(\Delta M / M \right)^2 \right> = %s$' % np.around(std_rel[i], 2), fontsize='small', horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, zorder=4)
            ax.grid(color='grey', linestyle=':', linewidth=0.5, zorder=1)
            ax.xaxis.set_ticks(ticks)
            ax.yaxis.set_ticks(ticks)
            if i == len(axes) - 1: ax.set_xlabel(r'True')
            ax.set_ylabel(r'Predicted')
            j = j+1
    plt.savefig('../data/models/' + modelName + 'graph/scatter_' + str(irun) + '.pdf', bbox_inches='tight', dpi=100)     


# # Let's load the data

# !ls ../data/models

# !ls ../data/

modelName = 'SDSS_URZ_arch_A/'
try:
    os.mkdir('../data/models/' + modelName)
    os.mkdir('../data/models/' + modelName + 'graph')
except:
    print('Model already exist')

data = h5py.File('../data/dataset_SDSS_URZ.h5py','r')

x_trainset = data['x_train'][()]
y_trainset = data['y_train'][()]
x_valset = data['x_val'][()]
y_valset = data['y_val'][()]
x_testset = data['x_test'][()]
y_testset = data['y_test'][()]

data.close()

ntrain, npix, _, nchannels = x_trainset.shape
nval, _, _, _ = x_valset.shape
ntest, _, _, _ = x_testset.shape

bins = np.geomspace(1, 100, 20) # Radial bins for the DM profile

nchannels

# +
train_sparsity = np.zeros((len(x_trainset), nchannels))
for i in range(len(x_trainset)):
    for j in range(nchannels):
        train_sparsity[i, j] = len(np.where(x_trainset[i,:,:,j] < 1)[0]) / (128*128)
        
test_sparsity = np.zeros((len(x_testset), nchannels))
for i in range(len(x_testset)):
    for j in range(nchannels):
        test_sparsity[i, j] = len(np.where(x_testset[i,:,:,j] < 1)[0]) / (128*128)
        
val_sparsity = np.zeros((len(x_valset), nchannels))
for i in range(len(x_valset)):
    for j in range(nchannels):
        val_sparsity[i, j] = len(np.where(x_valset[i,:,:,j] < 1)[0]) / (128*128)
# -

# ## Let's normalize the in/outputs

# +
# For the images we will put the images between 0 and 1 using the images of the trainset
#xmin = np.min(x_trainset)
#xmax = np.max(x_trainset)
#
#x_trainset = (x_trainset - xmin) / (xmax - xmin)
#x_testset  = (x_testset - xmin) / (xmax - xmin)
#x_valset   = (x_valset - xmin) / (xmax - xmin)

# +
# For the DM profile we will use the log of the mass

y_trainset = np.log(y_trainset)
y_testset  = np.log(y_testset)
y_valset   = np.log(y_valset)

# +
# For the DM profile we will put between 0 and 1 using the data of the trainset
#ymin = np.min(y_trainset, axis = 0)
#ymax = np.max(y_trainset, axis = 0)
#
#y_trainset = (y_trainset - ymin) / (ymax - ymin)
#y_testset  = (y_testset - ymin) / (ymax - ymin)
#y_valset   = (y_valset - ymin) / (ymax - ymin)
# -

# # Let's play

# ## Initialization of the model

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("YEAH! we have a GPU")
else:
    print("Not enough GPU hardware devices available")

# +
# network parameters
input_shape = (npix, npix, nchannels) # Input shape (#rows, #cols, #channels)
actFunction = 'relu'#tf.keras.layers.LeakyReLU(alpha=0.01) #'relu'

# Hidden layers dimensions
output_dim = y_trainset.shape[1]


# -

def initialization(input_shape, output_dim, actFunction):
    K.clear_session()
    tf.random.set_seed(28890)
    # build model
    model = Sequential()

    model.add(Conv2D(filters = 64, kernel_size = (5,5), strides = 2 , padding = 'same', input_shape = input_shape))
    model.add(Activation(actFunction))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25)) 
    model.add(BatchNormalization())

    model.add(Conv2D(filters = 128, kernel_size = (5,5), strides = 2 , padding = 'same'))
    model.add(Activation(actFunction))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25)) 
    model.add(BatchNormalization())

    model.add(Conv2D(filters = 128, kernel_size = (5,5), strides = 2 , padding = 'same'))
    model.add(Activation(actFunction))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25)) 
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(256, activation = actFunction))
    model.add(Dropout(0.25)) 
    model.add(BatchNormalization())

    model.add(Dense(128, activation = actFunction))
    model.add(Dropout(0.25)) 
    model.add(BatchNormalization())

    model.add(Dense(64, activation = actFunction))
    model.add(Dropout(0.25)) 
    model.add(BatchNormalization())

    model.add(Dense(output_dim, name = 'output', activation = 'linear'))
    return model
model = initialization(input_shape, output_dim, actFunction)
model.summary()

# +
# instantiate model
optimizer = optimizers.Adam(learning_rate = 1e-3, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)
model.compile(optimizer = optimizer, loss = 'mse', metrics=['mae','mse'])

# Callbacks
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights = True)


# +
from tf_explain.callbacks.smoothgrad import SmoothGradCallback

smooth = SmoothGradCallback(
            validation_data=(x_valset, y_valset),
            class_index=0,
            num_samples=2,
            noise=1.,
            output_dir='../data/models/' + modelName,
        )

# -

# ## Some check previous to the fit 
# - [x] Overfitting 1 batch

out_pred = model.predict(x_trainset[:32,:,:,:])
score = model.evaluate(x_trainset[:32,:,:,:], y_trainset[:32,:], verbose=1) 
print(score)

# +
# %%time
batch_size = 32
epochs     = 5000

history = model.fit(x_trainset[:batch_size,:,:,:], y_trainset[:batch_size,:],
                  epochs           = epochs,
                  verbose          = 0)
# -

out_pred = model.predict(x_trainset[:batch_size,:,:,:])
score = model.evaluate(x_trainset[:batch_size,:,:,:], y_trainset[:batch_size,:], verbose=1) 
print(score)

# ## Fitting the model

# +
np.random.seed(28890)

batch_size = 32
epochs     = 200

with h5py.File('../data/models/' + modelName + '/scores.h5', 'a') as scores:
    for i in range(0, 5):
        if 'run_' + str(i) not in scores.keys():
            # instantiate model
            model = initialization(input_shape, output_dim, actFunction)
            optimizer = optimizers.Adam(learning_rate = 1e-3, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)
            model.compile(optimizer = optimizer, loss = 'mse', metrics=['mae','mse'])

            # Let's compute a sample of all the train indeces witouth replacement
            train_indices = np.arange( len(x_trainset) )
            train_indices = np.random.choice(train_indices, len(x_trainset), replace = True) 

            x_trainset_cp = x_trainset[train_indices]
            y_trainset_cp = y_trainset[train_indices]

            print('Fitting model ' + str(i+1) + ' ...')
            history = model.fit(x_trainset_cp, y_trainset_cp,
                            epochs           = epochs,
                            verbose          = 0,
                            callbacks        = [es, smooth],
                            validation_data  = (x_valset, y_valset))
            score = model.evaluate(x_testset, y_testset,verbose=1) 
            print('Results obtained in the testset...')
            print(score)  
            print('Saving model ' + str(i+1) + ' ...')
            run = scores.create_group('run_' + str(i))

            run.create_dataset('train_loss', data = history.history['loss'])
            run.create_dataset('train_mae', data = history.history['mae'])
            run.create_dataset('train_mse', data = history.history['mse'])
            run.create_dataset('val_loss', data = history.history['val_loss'])
            run.create_dataset('val_mae', data = history.history['val_mae'])
            run.create_dataset('val_mse', data = history.history['val_mse'])
            model.save_weights('../data/models/' + modelName + 'weights_' + str(i) + '.hdf5')
            #MakePlots(model, x_testset, y_testset, modelName, i)
            K.clear_session()

# +
import PIL.Image
#from matplotlib import pylab as P

# From our repository.
import saliency.core as saliency
# -

nchannels

class_idx_str = 'class_idx_str'
def call_model_function(images, call_model_args=None, expected_keys=None):
    images = tf.convert_to_tensor(images)
    with tf.GradientTape() as tape:
        tape.watch( images )
        output_layer = model( tf.reshape(images, (2,128,128,3)) )
        output_layer = output_layer[0,4]
        gradients = np.array(tape.gradient(output_layer, images))
        return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}



# +
call_model_args = {class_idx_str: 5}
# Construct the saliency object. This alone doesn't do anthing.
gradient_saliency = saliency.GradientSaliency()

# Compute the vanilla mask and the smoothed mask.
vanilla_mask_3d = gradient_saliency.GetMask(x_testset[0:2], call_model_function, call_model_args)
smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(x_testset[0:2], call_model_function, call_model_args)
# -

vanilla_mask_3d.shape

plt.imshow(smoothgrad_mask_3d[0,:,:,0])

# +
img_idx = np.random.randint(0, y_testset.shape[0])
ch = 0
fig, axes = plt.subplots(4,5,figsize=(14,14), sharex=True, sharey=True, gridspec_kw={'hspace':-0.4, 'wspace':0.1})

axes[0,0].imshow(x_testset[img_idx,:,:,ch])
inax = axes[0,0].inset_axes([0,0.7,0.3,0.3])
inax.plot(bins, y_testset[img_idx,:], c = 'black')
inax.scatter(bins, model.predict(x_testset[img_idx].reshape(1,128,128,nchannels)), c = 'blue', marker = '.')
inax.set_xscale('log')
#inax.set_yscale('log')
inax.set_xlabel('')
inax.set_ylabel('')
inax.set_xticks([])
inax.set_yticks([])

for i in range(19):
    input_img = tf.reshape(tf.Variable(x_testset[img_idx,:,:,:], dtype=float, trainable = True), (1,128,128,nchannels))
    with tf.GradientTape() as tape:
        tape.watch(input_img)
        result = model(input_img)[0,i]
        
    grads = tape.gradient(result, input_img)     
    dgrad_abs = tf.math.abs(grads)
    dgrad_max = np.max(dgrad_abs, axis=3)[0]
    arr_min, arr_max  = np.min(dgrad_max), np.max(dgrad_max)
    grad_eval = (dgrad_max - arr_min) / (arr_max - arr_min + 1e-18)
    
    i = axes[(i+1)//5, (i+1)%5].imshow((grad_eval),cmap="jet")
    #fig.colorbar(i)
# -


