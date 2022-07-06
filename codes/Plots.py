# +
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

import os


# -

# # Some custom functions

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


def initialization():
    K.clear_session()
    tf.random.set_seed(28890)
    # build model
    model = Sequential()

    model.add(Conv2D(filters = 64, kernel_size = (5,5), strides = 2 , padding = 'same', input_shape = input_shape))
    model.add(Activation(actFunction))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25)) 
    model.add(BatchNormalization())

    model.add(Conv2D(filters = 128, kernel_size = (5,5), strides = 2 , padding = 'same', input_shape = input_shape))
    model.add(Activation(actFunction))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25)) 
    model.add(BatchNormalization())

    model.add(Conv2D(filters = 128, kernel_size = (5,5), strides = 2 , padding = 'same', input_shape = input_shape))
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

modelName = 'SDSS_I_HI_1_arch_A/'


data = h5py.File('../data/dataset_SDSS_I_HI_1.h5py','r')

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

y_trainset = np.log(y_trainset)
y_testset  = np.log(y_testset)
y_valset   = np.log(y_valset)

# +
# network parameters
input_shape = (npix, npix, nchannels) # Input shape (#rows, #cols, #channels)
actFunction = 'relu'#tf.keras.layers.LeakyReLU(alpha=0.01) #'relu'

# Hidden layers dimensions
output_dim        = y_trainset.shape[1]
model = initialization()
model.summary()
# -

# # Let's read the weights for each run

scores = h5py.File('../data/models/' + modelName + '/scores.h5', 'r')
scores.keys()

val_loss = np.zeros((len(scores), len(scores['run_0']['val_loss'])))
train_loss = np.zeros((len(scores), len(scores['run_0']['val_loss'])))
for i, sc in enumerate(scores):
    val_loss[i,:] = scores[sc]['val_loss']
    train_loss[i,:] = scores[sc]['train_loss']

# +
mean_val_loss = np.mean(val_loss, axis = 0)
std_val_loss = np.std(val_loss, axis = 0)
mean_train_loss = np.mean(train_loss, axis = 0)
std_train_loss = np.std(train_loss, axis = 0)

plt.plot(mean_val_loss, label = 'Val loss')
plt.fill_between(np.arange(0, len(scores['run_0']['val_loss'])), (mean_val_loss - std_val_loss), (mean_val_loss + std_val_loss), alpha = 0.5 )
plt.plot(mean_train_loss, label = 'Train loss', linestyle = ':')
plt.fill_between(np.arange(0, len(scores['run_0']['val_loss'])), (mean_train_loss - std_train_loss), (mean_train_loss + std_train_loss), alpha = 0.5 )
plt.legend()
plt.yscale('log')
plt.xscale('log')
# -

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

# +
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

# -

# Residuals
f, ax = plt.subplots()
ax.set_xlabel(r'$\Delta \log M$')
ax.set_ylabel(r'P($\Delta \log M$)')
for i in range(20):
    ax.hist(diff_log[:,i], bins=20, density=True, histtype='step', alpha=1, label=r'$r = %s \; \mathrm{kpc}$'%np.around(bins[i], 1))
#plt.legend(loc='upper left', fontsize='small')
ax.grid(color='grey', linestyle=':', linewidth=0.25)
plt.savefig('../data/models/' + modelName + 'graph/MeanResiduals.pdf')

# +
# MSE and MAE in log
f, ax = plt.subplots()

ax.set_xlabel(r'$r$ [kpc]')
ax.set_ylabel(r'$\Delta$')
ax.plot(np.around(bins, 1), mse_log_rel, label=r'$\left< (\frac{\Delta \mu}{\mu})^{2} \right>(r)$')
ax.plot(np.around(bins, 1), mae_log_rel, label=r'$\left< \left|\Delta \mu / \mu \right| \right>(r)$')
ax.grid(color='grey', linestyle=':', linewidth=0.25, which='both')
ax.set_xscale('log')
ax.legend()
plt.savefig('../data/models/' + modelName + 'graph/Mean_MSE_MAE_log.pdf')


# +
# MSE and MAE
f, ax = plt.subplots()

ax.set_xlabel(r'$r$ [kpc]')
ax.set_ylabel(r'$\Delta$')
ax.plot(np.around(bins, 1), mse_rel, label=r'$\left< (\frac{\Delta M}{M})^{2} \right>(r)$')
ax.plot(np.around(bins, 1), mae_rel, label=r'$\left< \left|\Delta M / M \right| \right>(r)$')
ax.grid(color='grey', linestyle=':', linewidth=0.25, which='both')
ax.set_xscale('log')
ax.legend()
plt.savefig('../data/models/' + modelName + 'graph/Mean_MSE_MAE.pdf')
# -

# # Deprecated

ResidualPlot(model, x_testset, y_testset, modelName, 49, save = False)

SigmaPlot(model, x_testset, y_testset, modelName, 49, save = False)

SigmaPlot(model, x_testset, y_testset, modelName, 49, save = False, log = False)

ScatterPlot(model, x_testset, y_testset, modelName, 49, save = False)


