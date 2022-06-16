# Let's load the needed libraries
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Let's load the data
data = h5py.File('../data/data_corrected.hdf5','r')
data.keys()

# # Let's split in 3 sets putting all the 3 images of the same subhalos in the same set

galsIds = list(data['subs/'].keys())
galsIds = galsIds[::3]


data['subs/102683_0'].keys()

ngals = len(galsIds)
channels = ['HI_mom1']
nchannels = len(channels)
npix = 128

ntrain = int(ngals * 70 / 100) # Number of galaxies in the trainset
nval   = int(ngals * 20 / 100) # Number of galaxies in the valset
ntest  = ngals - ntrain - nval # Number of galaxies in the testset

# +
np.random.seed(28890)
ind = np.random.choice(np.arange(0, ngals), ntrain, replace = False)

train_ind = ind[:ntrain]
val_ind   = ind[ntrain:(ntrain + nval)]
test_ind  = ind[(ntrain + nval):]

# +
x0_train = np.zeros((ntrain, npix, npix, nchannels))
y0_train = np.zeros((ntrain, 20))
x1_train = np.zeros((ntrain, npix, npix, nchannels))
y1_train = np.zeros((ntrain, 20))
x2_train = np.zeros((ntrain, npix, npix, nchannels))
y2_train = np.zeros((ntrain, 20))

for i in tqdm(range(ntrain)):
    subName0 = galsIds[i][:-2] + '_0'
    gal0 = data['subs/' + subName0]
    subName1 = galsIds[i][:-2] + '_1'
    gal1 = data['subs/' + subName1]
    subName2 = galsIds[i][:-2] + '_2'
    gal2 = data['subs/' + subName2]
    for j in range(nchannels):
        x0_train[i, :,:, j] = gal0[channels[j]][()]
        y0_train[i, :]      = gal0['dm_profile'][()]
        x1_train[i, :,:, j] = gal1[channels[j]][()]
        y1_train[i, :]      = gal1['dm_profile'][()]
        x2_train[i, :,:, j] = gal2[channels[j]][()]
        y2_train[i, :]      = gal2['dm_profile'][()]
        
x_train = np.vstack((x0_train, x1_train, x2_train))
y_train = np.vstack((y0_train, y1_train, y2_train))

# +
x0_val = np.zeros((nval, npix, npix, nchannels))
y0_val = np.zeros((nval, 20))
x1_val = np.zeros((nval, npix, npix, nchannels))
y1_val = np.zeros((nval, 20))
x2_val = np.zeros((nval, npix, npix, nchannels))
y2_val = np.zeros((nval, 20))

for i in tqdm(range(nval)):
    subName0 = galsIds[i][:-2] + '_0'
    gal0 = data['subs/' + subName0]
    subName1 = galsIds[i][:-2] + '_1'
    gal1 = data['subs/' + subName1]
    subName2 = galsIds[i][:-2] + '_2'
    gal2 = data['subs/' + subName2]
    for j in range(nchannels):
        x0_val[i, :,:, j] = gal0[channels[j]][()]
        y0_val[i, :]      = gal0['dm_profile'][()]
        x1_val[i, :,:, j] = gal1[channels[j]][()]
        y1_val[i, :]      = gal1['dm_profile'][()]
        x2_val[i, :,:, j] = gal2[channels[j]][()]
        y2_val[i, :]      = gal2['dm_profile'][()]
        
x_val = np.vstack((x0_val, x1_val, x2_val))
y_val = np.vstack((y0_val, y1_val, y2_val))

# +
x0_test = np.zeros((ntest, npix, npix, nchannels))
y0_test = np.zeros((ntest, 20))
x1_test = np.zeros((ntest, npix, npix, nchannels))
y1_test = np.zeros((ntest, 20))
x2_test = np.zeros((ntest, npix, npix, nchannels))
y2_test = np.zeros((ntest, 20))

for i in tqdm(range(ntest)):
    subName0 = galsIds[i][:-2] + '_0'
    gal0 = data['subs/' + subName0]
    subName1 = galsIds[i][:-2] + '_1'
    gal1 = data['subs/' + subName1]
    subName2 = galsIds[i][:-2] + '_2'
    gal2 = data['subs/' + subName2]
    for j in range(nchannels):
        x0_test[i, :,:, j] = gal0[channels[j]][()]
        y0_test[i, :]      = gal0['dm_profile'][()]
        x1_test[i, :,:, j] = gal1[channels[j]][()]
        y1_test[i, :]      = gal1['dm_profile'][()]
        x2_test[i, :,:, j] = gal2[channels[j]][()]
        y2_test[i, :]      = gal2['dm_profile'][()]
        
x_test = np.vstack((x0_test, x1_test, x2_test))
y_test = np.vstack((y0_test, y1_test, y2_test))
# -

dataset = h5py.File('../data/dataset_HI_1.h5py', 'w')
dataset.create_dataset('x_train', data = x_train)
dataset.create_dataset('y_train', data = y_train)
dataset.create_dataset('x_val', data = x_val)
dataset.create_dataset('y_val', data = y_val)
dataset.create_dataset('x_test', data = x_test)
dataset.create_dataset('y_test', data = y_test)
dataset.close()

# # Deprecated

ngals = len(data['subs'])
channels = ['SDSS_I']
nchannels = len(channels)
npix = 128

# +
x_array = np.zeros((ngals, npix, npix, nchannels))
y_array = np.zeros((ngals, 20))

for igal, gal in tqdm(enumerate(data['subs/'])):
    gal = data['subs/' + gal]
    for j in range(nchannels):
        x_array[igal, :,:, j] = gal[channels[j]][()]
        y_array[igal, :]      = gal['dm_profile'][()]
# -

dataset = h5py.File('../data/dataset_SDSS_I.h5py', 'w')
dataset.create_dataset('x_data', data = x_array)
dataset.create_dataset('y_data', data = y_array)
dataset.close()
