# # Some neccesary libraries and functions

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

# # Let's load the data

modelName = 'SDSS_I_arch_A_2/'
try:
    os.mkdir('../data/models/' + modelName)
except:
    print('Model already exist')

data = h5py.File('../data/dataset_SDSS_I.h5py','r')
x_data = data['x_data'][()]
y_data = data['y_data'][()]
data.close()

ngals, npix, _, nchannels = x_data.shape
ngals = int(ngals / 3) # We have 3 images per galaxy

# +
# Let's split into train and test sets

ntrain = int(80 * ngals / 100)
nval   = int(15 * ngals / 100)
ntest  = ngals - ntrain - nval

x_trainset = x_data[:(3 * ntrain)]
x_valset   = x_data[(3 * ntrain):(3 * ntrain + 3 * nval)]
x_testset  = x_data[(3 * ntrain + 3 * nval):]

y_trainset = y_data[:(3 * ntrain)]
y_valset   = y_data[(3 * ntrain):(3 * ntrain + 3 * nval)]
y_testset  = y_data[(3 * ntrain + 3 * nval):]
# -

bins = np.geomspace(1, 100, 20) # Radial bins for the DM profile

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
output_dim        = y_trainset.shape[1]


# -

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
model = initialization()
model.summary()

# instantiate model
optimizer = optimizers.Adam(learning_rate = 1e-3, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)
model.compile(optimizer = optimizer, loss = 'mse', metrics=['mae','mse'])

# ## Some check previous to the fit

out_pred = model.predict(x_trainset[:32,:,:,:])
score = model.evaluate(x_trainset[:32,:,:,:], y_trainset[:32,:], verbose=1) 
print(score)

# +
batch_size = 32
epochs     = 1500

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
epochs     = 100

for i in range(1, 4):
#i=0  
    # instantiate model
    model = initialization()
    optimizer = optimizers.Adam(learning_rate = 1e-3, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)
    model.compile(optimizer = optimizer, loss = 'mse', metrics=['mae','mse'])

    # Let's compute a sample of all the train indeces witouth replacement
    train_indices = np.arange( len(x_trainset) )
    train_indices = np.random.choice(train_indices, len(x_trainset), replace = True) 

    x_trainset_cp = x_trainset[train_indices]
    y_trainset_cp = y_trainset[train_indices]

    print('Fitting ' + str(i) + ' model ...')
    history = model.fit(x_trainset_cp, y_trainset_cp,
                    epochs           = epochs,
                    verbose          = 0,
                    #callbacks        = [es],
                    validation_data  = (x_valset, y_valset))
    score = model.evaluate(x_testset, y_testset,verbose=1) 
    print('Results obtained in the testset...')
    print(score)  
    print('Saving ' + str(i) + ' model ...')
    model.save_weights('../data/models/' + modelName + 'weights_' + str(i) + '.hdf5')
    K.clear_session()
# -

# # Plots

out_pred = model.predict(x_testset)

real_masses = np.exp(y_testset) # If exp
pred_masses = np.exp(out_pred)

diff = real_masses - pred_masses

avg     = np.average(abs(diff), 0)
avg_rel = np.average(abs(diff) / real_masses, 0)
std     = np.std(diff, 0)
std_rel = np.std(diff / real_masses, 0)
dex_abs = np.average(abs(np.log10(real_masses / pred_masses)), 0)
dex     = np.average(np.log10(real_masses / pred_masses), 0)

masses      = np.log(real_masses)
predictions = np.log(pred_masses)

# +
outputs = 20 # len(real_masses[0,:]) -1
rPts    = bins#np.array([2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
plt.cla()
plt.clf()
fig, axes          = plt.subplots(5, figsize=(5, 50), sharex=True)
mass_min, mass_max = int(np.amin([masses, predictions])), int(np.amax([masses, predictions])) + 1
ticks              = np.arange(mass_min, mass_max + 1)

j = 0
for i in range(outputs):
    if (i%5 == 0):
        ax = axes[j,]
        ax.plot((mass_min, mass_max), (mass_min, mass_max), color='black', ls='--', lw=1, zorder=2)
        ax.scatter(masses[:,i], predictions[:,i], s=0.001, zorder=3)
        ax.set_aspect('equal', 'box')
        ax.set_title(r'log $M_\mathrm{DM}(r \leq %s \mathrm{kpc})$' % np.around(rPts[i], 1))
        ax.text(0.05, 0.95, r'$\left< \left(\Delta M / M \right)^2 \right> = %s$' % np.around(std_rel[i], 2), fontsize='small', horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, zorder=4)
        ax.grid(color='grey', linestyle=':', linewidth=0.5, zorder=1)
        ax.xaxis.set_ticks(ticks)
        ax.yaxis.set_ticks(ticks)
        if i == len(axes) - 1: ax.set_xlabel(r'True')
        ax.set_ylabel(r'Predicted')
        j = j+1
#plt.savefig(folderName + '/scatter.pdf', bbox_inches='tight', dpi=100)
# -

diff = masses - predictions
plt.cla()
plt.clf()
plt.xlabel(r'$\Delta \log M$')
plt.ylabel(r'P($\Delta \log M$)')
for i in range(outputs):
    plt.hist(diff[:,i], bins=20, density=True, histtype='step', alpha=1, label=r'$r = %s \; \mathrm{kpc}$'%np.around(rPts[i], 1))
#plt.legend(loc='upper left', fontsize='small')
plt.grid(color='grey', linestyle=':', linewidth=0.25)
plt.tight_layout()
#plt.savefig(folderName + '/residuals.pdf')

plt.cla()
plt.clf()
plt.xlabel(r'$r$ [kpc]')
plt.ylabel(r'$\Delta$')
plt.plot(np.around(rPts, 1), avg_rel, label=r'$\left< \left|\Delta M / M \right| \right>(r)$')
plt.plot(np.around(rPts, 1), std_rel, label=r'$\left< \left(\Delta M / M \right)^2 \right>(r)$')
plt.grid(color='grey', linestyle=':', linewidth=0.25, which='both')
plt.xscale('log')
plt.legend()
plt.tight_layout()
#plt.savefig(folderName + '/sigma.pdf')


