# +
from tensorflow.keras.layers import Input, Dense, Dropout, AveragePooling2D, Flatten, Conv2D
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
#from vis.visualization import visualize_saliency

import numpy as np
import matplotlib.pyplot as plt
import h5py
from random import randint
import os
# -

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("YEAH! we have a GPU")
else:
    print("Not enough GPU hardware devices available")

# # Some custom functions

bins = np.geomspace(1, 100, 20) # Radial bins for the DM profile


def ReadData(datasetName, modelName, irun = None):
    # Let's read the data ------------------------------------------
    data = h5py.File('../data/dataset_' + datasetName + '.h5py','r')
    x_testset = data['x_test'][()]
    y_testset = data['y_test'][()]
    data.close()
    ntest, npix, _, nchannels = x_testset.shape
    bins = np.geomspace(1, 100, 20) # Radial bins for the DM profile
    y_testset  = np.log(y_testset)
    # ----------------------------------------------------------------
    
    # Let's read the scores obtanied in the training ---------------------------
    scores = h5py.File('../data/models/' + modelName + '/scores.h5', 'r')
    val_loss = np.zeros((len(scores), 200))
    train_loss = np.zeros((len(scores), 200))
    for i, sc in enumerate(scores):
        val_loss[i,:len(scores[sc]['val_loss'])] = scores[sc]['val_loss']
        train_loss[i,:len(scores[sc]['train_loss'])] = scores[sc]['train_loss']
    # --------------------------------------------------------------------------
        
    # Let's initialize the network and read the weights -------------------------
    # network parameters
    input_shape = (npix, npix, nchannels) # Input shape (#rows, #cols, #channels)
    actFunction = 'relu'#tf.keras.layers.LeakyReLU(alpha=0.01) #'relu'

    # Hidden layers dimensions
    output_dim = y_testset.shape[1]
    model = initialization(input_shape, output_dim, actFunction)
            
    if irun is None:
        mse = []
        test_pred = np.zeros((len(scores), len(x_testset), 20))        
        for i in range(len(scores)):
            optimizer = optimizers.Adam(learning_rate = 1e-3, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)
            model.compile(optimizer = optimizer, loss = 'mse', metrics=['mae','mse'])
            model.load_weights('../data/models/' + modelName + '/weights_' + str(i) + '.hdf5')
            mse.append(model.evaluate(x_testset, y_testset,verbose=0)[0])
            test_pred[i,:,:] = model.predict(x_testset) 
            
        mean_pred = np.mean(test_pred, axis=0)
        std_pred = np.std(test_pred, axis=0)

        real_masses = np.exp(y_testset) # If exp
        pred_masses = np.exp(mean_pred)
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
    else:
        mse = []
        test_pred = np.zeros((1, len(x_testset), 20))        
        
        optimizer = optimizers.Adam(learning_rate = 1e-3, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)
        model.compile(optimizer = optimizer, loss = 'mse', metrics=['mae','mse'])
        model.load_weights('../data/models/' + modelName + '/weights_' + str(irun) + '.hdf5')
        mse.append(model.evaluate(x_testset, y_testset,verbose=0)[0])
        test_pred[0,:,:] = model.predict(x_testset)
        
        mean_pred = np.mean(test_pred, axis=0)
        std_pred = 0

        real_masses = np.exp(y_testset) # If exp
        pred_masses = np.exp(mean_pred)
        diff = real_masses - pred_masses

        log_real_masses = np.log(real_masses)
        log_pred_masses= np.log(pred_masses)
        diff_log = log_real_masses - log_pred_masses

        mse = np.mean(diff**2, 0)
        std = 0
        mse_rel = np.mean((diff / real_masses) ** 2, 0)
        std_rel = 0
        mae = np.mean(np.abs(diff), 0)
        mae_std = 0
        mae_rel = np.mean(np.abs(diff / real_masses), 0)
        std_rel = 0

        mse_log = np.mean(diff_log**2, 0)
        std_log = 0
        mse_log_rel = np.mean((diff_log / log_real_masses) ** 2, 0)
        std_log_rel = 0
        mae_log = np.mean(np.abs(diff_log), 0)
        mae_log_std = 0
        mae_log_rel = np.mean(np.abs(diff_log / log_real_masses), 0)
        std_log_rel = 0
    # ----------------------------------------------------------------------------------------------------
    
    # Let's create a random regression for comparison
    
    rand_pred = np.random.normal(np.mean(y_testset,axis = 0), np.std(y_testset,axis = 0), y_testset.shape )

    rand_masses = np.exp(rand_pred)
    rand_diff = real_masses - rand_masses

    log_rand_masses= np.log(rand_masses)
    rand_diff_log = log_real_masses - log_rand_masses

    rand_mse = np.mean(rand_diff**2, 0)
    rand_std = np.std(rand_diff**2, 0)
    rand_mse_rel = np.mean((rand_diff / real_masses) ** 2, 0)
    rand_std_rel = np.std((rand_diff / real_masses) ** 2, 0)
    rand_mae = np.mean(np.abs(rand_diff), 0)
    rand_mae_std = np.std(np.abs(rand_diff), 0)
    rand_mae_rel = np.mean(np.abs(rand_diff / real_masses), 0)
    rand_std_rel = np.std(np.abs(rand_diff / real_masses), 0)

    rand_mse_log = np.mean(rand_diff_log**2, 0)
    rand_std_log = np.std(rand_diff_log**2, 0)
    rand_mse_log_rel = np.mean((rand_diff_log / log_real_masses) ** 2, 0)
    rand_std_log_rel = np.std((rand_diff_log / log_real_masses) ** 2, 0)
    rand_mae_log = np.mean(np.abs(rand_diff_log), 0)
    rand_mae_log_std = np.std(np.abs(rand_diff_log), 0)
    rand_mae_log_rel = np.mean(np.abs(rand_diff_log / log_real_masses), 0)
    rand_std_log_rel = np.std(np.abs(rand_diff_log / log_real_masses), 0)
    
    return model, x_testset, y_testset, mse_log_rel, mae_log_rel, diff_log, mse_rel, mae_rel, diff, train_loss, val_loss, rand_mse_log_rel, rand_mae_log_rel, rand_diff_log, rand_mse_rel, rand_mae_rel, rand_diff


def ResidualPlot(model, x_testset, y_testset, modelName, irun, save = False):
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
    f, ax = plt.subplots()
    ax.set_xlabel(r'$\Delta \log M$')
    ax.set_ylabel(r'P($\Delta \log M$)')
    for i in range(20):
        ax.hist(diff_log[:,i], bins=20, density=True, histtype='step', alpha=1, label=r'$r = %s \; \mathrm{kpc}$'%np.around(bins[i], 1))
    #plt.legend(loc='upper left', fontsize='small')
    ax.grid(color='grey', linestyle=':', linewidth=0.25)
    if save: plt.savefig('../data/models/' + modelName + 'graph/residuals_' + str(irun) +'.pdf')
    return ax


def SigmaPlot(model, x_testset, y_testset, modelName, irun, save = False, log = True):
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
    
    f, ax = plt.subplots()
    if log:
        # MSE and MAE in log

        ax.set_xlabel(r'$r$ [kpc]')
        ax.set_ylabel(r'$\Delta$')
        ax.plot(np.around(bins, 1), mse_log_rel, label=r'$\left< (\frac{\Delta \mu}{\mu})^{2} \right>(r)$')
        ax.plot(np.around(bins, 1), mae_log_rel, label=r'$\left< \left|\Delta \mu / \mu \right| \right>(r)$')
        ax.grid(color='grey', linestyle=':', linewidth=0.25, which='both')
        ax.set_xscale('log')
        ax.legend()
        if save: plt.savefig('../data/models/' + modelName + 'graph/MSE_MAE_log_' + str(irun) + '.pdf')
    else:
        # MSE and MAE
        ax.set_xlabel(r'$r$ [kpc]')
        ax.set_ylabel(r'$\Delta$')
        ax.plot(np.around(bins, 1), mse_rel, label=r'$\left< (\frac{\Delta M}{M})^{2} \right>(r)$')
        ax.plot(np.around(bins, 1), mae_rel, label=r'$\left< \left|\Delta M / M \right| \right>(r)$')
        ax.grid(color='grey', linestyle=':', linewidth=0.25, which='both')
        ax.set_xscale('log')
        ax.legend()
        if save: plt.savefig('../data/models/' + modelName + 'graph/MSE_MAE_' + str(irun) + '.pdf')

    return ax


def ScatterPlot(model, x_testset, y_testset, modelName,irun, save = False):
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
    
    # Scatter-plots
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
    if save: plt.savefig('../data/models/' + modelName + 'graph/scatter_' + str(irun) + '.pdf', bbox_inches='tight', dpi=100)     
    return ax


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


# # Let's read the data

# !ls ../data/models

# !ls ../data/

modelName = 'SDSS_URZ_HI_012_arch_A/'
datasetName = 'SDSS_URZ_HI_012'


model, x_testset, y_testset, mse_log_rel, mae_log_rel, diff_log, mse_rel, mae_rel, diff, train_loss, val_loss, rand_mse_log_rel, rand_mae_log_rel, rand_diff_log, rand_mse_rel, rand_mae_rel, rand_diff = ReadData(datasetName, modelName, 5)

# # Let's make some plots

# +
mean_val_loss = np.mean(val_loss, axis = 0)
std_val_loss = np.std(val_loss, axis = 0)
mean_train_loss = np.mean(train_loss, axis = 0)
std_train_loss = np.std(train_loss, axis = 0)

plt.plot(mean_val_loss, label = 'Val loss')
plt.fill_between(np.arange(0, 200), (mean_val_loss - std_val_loss), (mean_val_loss + std_val_loss), alpha = 0.5 )
plt.plot(mean_train_loss, label = 'Train loss', linestyle = ':')
plt.fill_between(np.arange(0, 200), (mean_train_loss - std_train_loss), (mean_train_loss + std_train_loss), alpha = 0.5 )
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Epochs')
plt.ylabel('Loss')
#plt.savefig('../data/models/' + modelName + 'graph/loss_history.pdf')
# -

# Residuals
f, ax = plt.subplots()
ax.set_xlabel(r'$\Delta \log M$')
ax.set_ylabel(r'P($\Delta \log M$)')
for i in range(20):
    ax.hist(diff_log[:,i], bins=20, density=True, histtype='step', alpha=1, label='r = {:.2e}'.format(np.around(bins[i], 1)) +  ' $\mathrm{kpc}$')
    ax.hist(rand_diff_log[:,i], bins=20, density=True, histtype='step', alpha=1, label=r'Random', color = 'black', linestyle = ':')
#plt.legend(loc='upper left', fontsize='small')
ax.grid(color='grey', linestyle=':')
#plt.savefig('../data/models/' + modelName + 'graph/MeanResiduals.pdf')

# +
# MSE and MAE in log
f, ax = plt.subplots()

ax.set_xlabel(r'$r$ [kpc]')
ax.set_ylabel(r'$\Delta$')
ax.plot(np.around(bins, 1), mse_log_rel, label=r'$\left< (\frac{\Delta \mu}{\mu})^{2} \right>(r)$')
ax.plot(np.around(bins, 1), mae_log_rel, label=r'$\left< \left|\Delta \mu / \mu \right| \right>(r)$')
ax.plot(np.around(bins, 1), rand_mse_log_rel, label=r'Random', color = 'black', linestyle =':')
ax.plot(np.around(bins, 1), rand_mae_log_rel, color = 'black', linestyle =':')
ax.grid(color='grey', linestyle=':', which='both')
ax.set_xscale('log')
ax.legend()
#plt.savefig('../data/models/' + modelName + 'graph/Mean_MSE_MAE_log.pdf')


# +
# MSE and MAE
f, ax = plt.subplots()

ax.set_xlabel(r'$r$ [kpc]')
ax.set_ylabel(r'$\Delta$')
ax.plot(np.around(bins, 1), mse_rel, label=r'$\left< (\frac{\Delta M}{M})^{2} \right>(r)$')
ax.plot(np.around(bins, 1), mae_rel, label=r'$\left< \left|\Delta M / M \right| \right>(r)$')
ax.plot(np.around(bins, 1), rand_mse_rel, label=r'Random', color = 'black', linestyle =':')
ax.plot(np.around(bins, 1), rand_mae_rel, color = 'black', linestyle =':')
ax.legend()
ax.set_xscale('log')
ax.grid(color='grey', linestyle=':', which='both')
#plt.savefig('../data/models/' + modelName + 'graph/Mean_MSE_MAE.pdf')

# +
#img_idx = randint(0, y_testset.shape[0])
ch = 5
fig, axes = plt.subplots(4,5,figsize=(14,14), sharex=True, sharey=True, gridspec_kw={'hspace':-0.4, 'wspace':0.1})

axes[0, 0].imshow(x_testset[img_idx,:,:,ch])

for i in range(19):
    input_img = tf.reshape(tf.Variable(x_testset[img_idx,:,:,:], dtype=float, trainable = True), (1,128,128,6))
    with tf.GradientTape() as tape:
        tape.watch(input_img)
        result = model(input_img)[0,i]
        
    grads = tape.gradient(result, input_img)     
    dgrad_abs = tf.math.abs(grads)
    dgrad_max = grads.numpy()[0,:,:,ch]#np.max(dgrad_abs, axis=3)[0]
    arr_min, arr_max  = np.min(dgrad_max), np.max(dgrad_max)
    grad_eval = (dgrad_max - arr_min) / (arr_max - arr_min + 1e-18)
    
    i = axes[(i+1)//5, (i+1)%5].imshow(grad_eval,cmap="jet")
    #fig.colorbar(i)

# +
from random import randint

def get_feature_maps(model, layer_id, input_image):
    model_ = Model(inputs=[model.input], 
                   outputs=[model.layers[layer_id].output])
    return model_.predict(np.expand_dims(input_image, 
                                         axis=0))[0,:,:,:].transpose((2,0,1))

def plot_features_map(img_idx=None, layer_idx=[0, 5, 10], 
                      x_test=x_testset, ytest=y_testset, cnn=model):
    if img_idx == None:
        img_idx = randint(0, ytest.shape[0])
    input_image = x_test[img_idx]
    
    fig, ax = plt.subplots(3,3,figsize=(10,10))
    ax[0][0].imshow(input_image)
    ax[0][0].set_title('original img id {}'.format(img_idx))
    
    feature_map = get_feature_maps(cnn, layer_idx[0], input_image)
    for j in range(2):
        ax[0][j+1].imshow(feature_map[:,:,j])
        ax[0][j+1].set_title('layer {} - {}'.format(layer_idx[0], cnn.layers[layer_idx[0]].get_config()['name']))
    
    feature_map = get_feature_maps(cnn, layer_idx[1], input_image)
    for j in range(3):
        ax[1][j].imshow(feature_map[:,:,j])
        ax[1][j].set_title('layer {} - {}'.format(layer_idx[1], cnn.layers[layer_idx[1]].get_config()['name']))
        
    feature_map = get_feature_maps(cnn, layer_idx[2], input_image)
    for j in range(3):
        ax[2][j].imshow(feature_map[:,:,j])
        ax[2][j].set_title('layer {} - {}'.format(layer_idx[2], cnn.layers[layer_idx[2]].get_config()['name']))
        
    return img_idx

img_idx = plot_features_map()
# -

# # Deprecated

ResidualPlot(model, x_testset, y_testset, modelName, 49, save = False)

SigmaPlot(model, x_testset, y_testset, modelName, 49, save = False)

SigmaPlot(model, x_testset, y_testset, modelName, 49, save = False, log = False)

ScatterPlot(model, x_testset, y_testset, modelName, 49, save = False)


