# # Importing the needed libraries

import os
import sys
import time
import argparse
import requests
import contextlib
from tqdm import tqdm
import tempfile
import h5py
import atexit
import numpy as np
from illustris_python.groupcat import loadSingle, loadHeader
import illustris_python as il
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mpl3
from numpy.linalg import eig
from numpy.linalg import eigh
from scipy.optimize import curve_fit, root
from scipy.spatial.transform import Rotation as R
from scipy.stats import binned_statistic
from matplotlib.gridspec import GridSpec


G     = 4.3e-6 # Grav. constant [kPc/M_{sun} (km/s)^2]
H0    = 67.74 # Hubble Constant [km/s / Mpc]
h     = H0 / 100 
rho_c = 3*(H0**2)/(8*np.pi*G*1e-3) # Critical density [M_{sun}/Mpc**3]
rho_c = rho_c * (1e-3 ** 3) #2.7754 * 1e2 * (H0/100)**2 # Critical density [M_{sun}/Kpc**3]
Nfields = 9
M_dm    = 7.5e6 # M_sun
headers = {"api-key": '81b7c70637fa8b110e6b9f236ea07c37'}


# # Some custom functions

def get(path, params=None, folderName=''):
    '''
    Illustris function
    '''
    # make HTTP GET request to path
    r = requests.get(path, params=params, headers=headers)

    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically

    if 'content-disposition' in r.headers:
        filename = r.headers['content-disposition'].split("filename=")[1]
        if filename.endswith('.hdf5'):
            file_access_property_list = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
            file_access_property_list.set_fapl_core(backing_store=False)
            file_access_property_list.set_file_image(r.content)
            
            file_id_args = {
                'fapl': file_access_property_list,
                'flags': h5py.h5f.ACC_RDONLY,
                'name': next(tempfile._get_candidate_names()).encode()
            }
            
            h5_file_args = {'backing_store': False, 'driver': 'core', 'mode': 'r'}
            with contextlib.closing(h5py.h5f.open(**file_id_args)) as file_id:
                with h5py.File(file_id, **h5_file_args) as h5_file:
                    #return np.array(h5_file['grid'])
                    if 'grid' in h5_file.keys(): return np.array(h5_file['grid'])
                    else:
                        results = []
                        for k in h5_file.keys():
                            for sk in h5_file[k].keys():
                                results.append(np.array(h5_file[k][sk]))
                        return results
        else:
            with open(folderName + filename, 'wb') as f:
                f.write(r.content)
            return filename # return the filename string
    return r



def get1(path, name, params=None):
    '''
    Illustris function
    '''
    # make HTTP GET request to path
    headers = {"api-key":"81b7c70637fa8b110e6b9f236ea07c37"}
    r = requests.get(path, params=params, headers=headers)
    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()
    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically
    if 'content-disposition' in r.headers:
        filename = r.headers['content-disposition'].split("filename=")[1]
        with open(name + '.hdf5', 'wb') as f:
            f.write(r.content)
        return name + '.hdf5' # return the filename string
    return NULL


def compute_mass_profile(gid, center):
    '''
    MIHAEL FUNCTION: compute the dark matter mass enclosed in 20 radii
    from 1 to 100 kPc
    
    Parameters
    ----------
    gid : int 
        GroupID
    center : list
        (x,y,z) Position of the group
    
    Returns
    -------
    
    NP Array
        Array with the dark matter mass enclosed in 20 radii from 1 to 100 kPc
    '''
    dm = il.snapshot.loadHalo('/home/tnguser/sims.TNG/TNG100-1/output/', 99, gid, 'dm', fields=['Coordinates'])
    dm = np.where(dm > 32500, dm - 75000, dm)
    dm = np.where(dm < -32500, dm + 75000, dm)
    center = np.where(center > 32500, center - 75000, center)
    center = np.where(center < -32500, center + 75000, center)
    dm = dm - center
    dist = []
    for d in dm:
        D = np.sqrt(sum([c**2 for c in d]))
        if D < 100: dist.append(D)
    R_bins = np.geomspace(1, 100, 20)
    M = np.array([len(np.where(np.array(dist) < R)[0]) * M_dm for R in R_bins])
    return M


def compute_total_mass_profile(Rmax, Rmin, Nm, sub_meta, url):
    '''
    Computes the dark matter, stars and gas mass enclosed in Nm radii
    from Rmin to Rmax kPc
    
    Parameters
    ----------
    Rmin, Rmax : float 
        Min and Max radii
    Nm : int
        Number of radial bins
    sub_meta : str
        Illustris information of the subhalo
    url : str
        Url to the Illustris server
    
    Returns
    -------
    
    List
        List of 4 Arrays corresponding to the radial bins and the 
        dark matter, stars and gas mass enclosed in Nm radii from 
        Rmin to Rmax kPc
    '''
    center = np.array([sub_meta['pos_x'], sub_meta['pos_y'], sub_meta['pos_z']])
    particles  = get(url + 'cutout.hdf5', {'dm':'Coordinates',
                                                'gas':'Coordinates,Masses',
                                                'stars':'Coordinates,Masses'
                                               })
    
    dm = particles[2] - center
    dm = np.where(dm > 32500, dm - 75000, dm)
    dm = np.where(dm < -32500, dm + 75000, dm)
    
    dist_dm = []
    for d in dm:
        D = np.sqrt(sum([c**2 for c in d]))
        dist_dm.append(D)
    
    m_gas = particles[1] * 1e10/h
    gas = particles[0] - center
    gas = np.where(gas > 32500, gas - 75000, gas)
    gas = np.where(gas < -32500, gas + 75000, gas)
    
    dist_gas = []
    for d in gas:
        D = np.sqrt(sum([c**2 for c in d]))
        dist_gas.append(D)

    m_stars = particles[4] * 1e10/h
    stars = particles[3] - center
    stars = np.where(stars > 32500, stars - 75000, stars)
    stars = np.where(stars < -32500, stars + 75000, stars)
    
    dist_stars = []
    for d in stars:
        D = np.sqrt(sum([c**2 for c in d]))
        dist_stars.append(D)
            
    R_bins = np.geomspace(Rmin, Rmax, Nm)
    
    p_dm    = np.array([len(np.where(np.array(dist_dm) < R)[0]) * M_dm for R in R_bins])
    p_stars = np.array([sum(m_stars[np.where(np.array(dist_stars) < R)[0]]) for R in R_bins])
    p_gas   = np.array([sum(m_gas[np.where(np.array(dist_gas) < R)[0]]) for R in R_bins])
    return R_bins, p_dm, p_stars, p_gas


# +
def compute_rot_mat_inertia(coordinates, masses, Rmin=0, Rmax=20):
    '''
    MIHAEL FUNCTION: computes the intertia momenta of a subhalo with ID sid
    
    Parameters
    ----------
    
    
    Returns
    -------
    
    Matrix
        Rotation matrix for align the intertia momenta with the z-axis
    '''
    
    dist = np.linalg.norm(coordinates, axis=1)
    indices1 = np.argwhere(dist < Rmin)
    indices2 = np.argwhere(dist > Rmax)
    indices = np.concatenate((indices1, indices2))
    distances = np.delete(dist, indices)
    coordinates = np.delete(coordinates, indices, axis=0)
    masses = np.delete(masses, indices)
    
    I = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            if i == j: I[i][j] = np.sum(masses * (distances**2 - coordinates[:,i] * coordinates[:,j]))
            else: I[i][j] = np.sum(masses * (- coordinates[:,i] * coordinates[:,j]))
    
    I_eign = np.linalg.eigh(I)
    L = I_eign[1][2]
    #print(I)
    #print(L / np.linalg.norm(L))
    
    rot, _ = R.align_vectors([L, np.cross(L, [1,0,0])], [[0,0,1],[0,1,0]])
    return rot.as_matrix(), L


# -

def compute_rot_mat_angMom(coordinates, velocities, masses, Rmin = 0, Rmax = 20):
    
    dist = np.linalg.norm(coordinates, axis=1)
    indices1 = np.argwhere(dist < Rmin)
    indices2 = np.argwhere(dist > Rmax)
    indices = np.concatenate((indices1, indices2))
    distances = np.delete(dist, indices)
    
    coordinates = np.delete(coordinates, indices, axis = 0)
    masses = np.delete(masses, indices)
    velocities = np.delete(velocities, indices, axis = 0)
    
    L = (np.cross(coordinates, velocities).T * np.array(masses)).T
    Lmean = np.mean(L, axis=0)
    #print(Lmean / np.linalg.norm(Lmean))
    
    rot, _ = R.align_vectors([Lmean, np.cross(Lmean, [1,0,0])], [[0,0,1],[1,0,0]])
    return rot.as_matrix(), Lmean


# +
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform

class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, 'arrow3D', _arrow3D)
# -

# # Looking for the subhalos

# +
N     = 40000 # Number of samples.
Nm    = 20 #  'Number of radii
Rmin  = 1 # Maximal radius.
Rmax  = 100 # 'Maximal radius.
o     = 0 # Subhalo offset.
p     = 254 # Number of pixles.
D     = 200 # Image physical extent (in kpc).

Mmin  = 1e11 # Minimum total mass.
Mmax  = 1e13 # Maximum total mass.
Mdmin = 1e9 # Minimum dark matter mass in half radius.
Mdmax = 1e13 # Maximum dark matter mass in half radius.
Mgmin = 1e8 # Minimum gas mass.
Mgmax = 1e13 # Maximum gas mass.
Msmin = 1e10 # Minimum stellar mass.
Msmax = 1e12 # Maximum stellar mass.

sim   = 'TNG100-1' # Name of simulation run.
z     = 99  # Snapshot number.
myBasePath = '../sims.TNG/' + sim +'/output/'

mass_min      = (Mmin / h) * 1e-10 # Minimum total mass
mass_max      = (Mmax / h) * 1e-10 # Maximum total mass
dm_mass_min   = (Mdmin / h) * 1e-10 # Minimum total dm mass
dm_mass_max   = (Mdmax / h) * 1e-10 # Maximum total dm mass
gas_mass_min  = (Mgmin / h) * 1e-10 # Minimum total gas mass
gas_mass_max  = (Mgmax / h) * 1e-10 # Maximum total gas mass
star_mass_min = (Msmin / h) * 1e-10 # Minimum total star mass
star_mass_max = (Msmax / h) * 1e-10 # Maximum total star mass


subhalos_url = 'http://www.tng-project.org/api/' + sim + '/snapshots/' + str(z) + '/subhalos'
url          = subhalos_url
subhalos     = get(subhalos_url, {'limit': N, 'offset': o,
                                #'mass__gt': mass_min, 'mass__lt': mass_max,                                     
                                #'massinhalfrad_dm__gt':dm_mass_min,'massinhalfrad_dm__lt':dm_mass_max, 
                                #'mass_gas__gt': gas_mass_min, 'mass_gas__lt': gas_mass_max,
                                'mass_stars__gt': star_mass_min, #'mass_stars__lt': star_mass_max,
                                #'filterFlag': True, 'parent':0, 
                                #'sfr__gt':0.1,
                                'subhaloflag__lt':1})

nsubhalos = len(subhalos['results'])
nsubhalos
# -

# # Analyzing each individual subhalo (ie galaxy)

data = h5py.File('../data/gals_properties.h5', 'a')

list(data.keys())

try:
    flag_MainProps = True
    old_MainProps = data['MainProps'][()]
except:
    flag_MainProps = False

# +
# Initialization of properties 
properties = np.zeros((nsubhalos, 18))

# 0: ID
# 1: central (1 if central, 0 if not)
# 2: SubMass [Msun]
# 3: SubSFR
# 4: SubHMR [kPc]
# 5: x [kPc]
# 6: y [kPc]
# 7: z [kPc]
# 8: vx [km/s]
# 9: vy [km/s]
# 10: vz [km/s]
# 11: SubVmax [km/s]
# 12: SubVmaxR [kPc]
# 13: SubHMRG [kPc] Comoving radius containing half of the stars mass of this Subhalo 
# 14: costheta. Cosine of the angle between the angular momenta and the main axis
                # of the inertia tensor.
# 15: kappa_AM
# 16: kappa_IT
# 17: analysis_flag: 1 If everything ended well, 0 otherwise

#i = 2
for i in tqdm(range(1,nsubhalos)):
    ids = subhalos['results'][i]['id']
    try:
        gr = data.create_group('SubID_' + str(ids))
        flag_gr = True
    except:
        print('Subhalo already exists')
        flag_gr = False

    if flag_gr:
        # Let's load the data of the subhalos
        sub_meta = get(subhalos['results'][i]['url'])
        # --------------------------------------------------------

        # Let's save the main properties  ------------------------           
        properties[i, 0] = ids   
        gid = sub_meta['grnr']
        if gid == ids:
            properties[i, 1] = 1
        properties[i, 2] = sub_meta['mass'] * 1e10 / h
        properties[i, 3] = sub_meta['sfr']
        properties[i, 4] = sub_meta['halfmassrad'] / h
        properties[i, 5] = sub_meta['pos_x'] / h
        properties[i, 6] = sub_meta['pos_y'] / h
        properties[i, 7] = sub_meta['pos_z'] / h
        properties[i, 8] = sub_meta['vel_x']
        properties[i, 9] = sub_meta['vel_y']
        properties[i, 10] = sub_meta['vel_z']
        properties[i, 11] = sub_meta['vmax']
        properties[i, 12] = sub_meta['vmaxrad'] / h
        properties[i, 13] = sub_meta['halfmassrad_stars'] / h 
        # --------------------------------------------------------

        # Let's estimate properties with the particles of the subhalos
        print('Starting the estimation of properties with subhalo particles for galaxy ' + str(ids))
        #try:
        
        sub_data_url = subhalos['results'][i]['url'] + 'vis.hdf5'
        center_sub   = properties[i, 5:8]
        velocity     = properties[i, 8:11]

        center_sub = np.where(center_sub > 32500, center_sub - 75000, center_sub)
        center_sub = np.where(center_sub < -32500, center_sub + 75000, center_sub)

        stars_c = get(subhalos['results'][i]['url'] + 'cutout.hdf5', {'stars':'Coordinates'})[0] / h
        stars_v = get(subhalos['results'][i]['url'] + 'cutout.hdf5', {'stars':'Velocities'})[0]
        stars_m = get(subhalos['results'][i]['url'] + 'cutout.hdf5', {'stars':'Masses'})[0] * 1e10 / h
        try:
            gas_c   = get(subhalos['results'][i]['url'] + 'cutout.hdf5', {'gas':'Coordinates'})[0] / h
            gas_v   = get(subhalos['results'][i]['url'] + 'cutout.hdf5', {'gas':'Velocities'})[0]
            gas_m   = get(subhalos['results'][i]['url'] + 'cutout.hdf5', {'gas':'Masses'})[0] * 1e10 / h
            flag_gas = True
        except:
            print('Galaxy ' + str(ids) + ' have no gas')
            flag_gas = False
        dm_c    = get(subhalos['results'][i]['url'] + 'cutout.hdf5', {'dm':'Coordinates'})[0] / h
        dm_v    = get(subhalos['results'][i]['url'] + 'cutout.hdf5', {'dm':'Velocities'})[0]

        print('Starting the estimation of properties with subhalo particles for galaxy ' + str(ids))

        # Let's move the coordinates if they are near the border
        stars_c = np.where(stars_c > 32500, stars_c - 75000, stars_c)
        stars_c = np.where(stars_c < -32500, stars_c + 75000, stars_c)
        if flag_gas:
            gas_c = np.where(gas_c > 32500, gas_c - 75000, gas_c)
            gas_c = np.where(gas_c < -32500, gas_c + 75000, gas_c)
        dm_c = np.where(dm_c > 32500, dm_c - 75000, dm_c)
        dm_c = np.where(dm_c < -32500, dm_c + 75000, dm_c)
        # --------------------------------------------------------

        # Let's move the coordinates to be center on the subhalo
        stars_c = stars_c - center_sub
        if flag_gas:
            gas_c   = gas_c - center_sub
        dm_c    = dm_c - center_sub

        stars_v = stars_v - velocity
        if flag_gas:
            gas_v   = gas_v - velocity
        dm_v    = dm_v - velocity
        # --------------------------------------------------------

        # Let's Compute the distance of each DM particle to the center and sum in radial bins
        dist = []
        for d in dm_c:
            D = np.sqrt(sum([c**2 for c in d]))
            if D < 100: dist.append(D)
        R_bins = np.geomspace(1, 100, 20)
        M = np.array([len(np.where(np.array(dist) < R)[0]) * M_dm for R in R_bins])
        # --------------------------------------------------------

        # Let's Compute the distance of each star particle to the center and sum in radial bins
        dist = []
        for d in stars_c:
            D = np.sqrt(sum([c**2 for c in d]))
            if D < 100: dist.append(D)
        M_stars = np.array([np.sum( stars_m[np.where(np.array(dist) < R)[0]] ) for R in R_bins])
        # --------------------------------------------------------

        # Compute the distance of each particle to the center and sum in radial bins
        if flag_gas:
            dist = []
            for d in gas_c:
                D = np.sqrt(sum([c**2 for c in d]))
                if D < 100: dist.append(D)
            M_gas = np.array([np.sum( gas_m[np.where(np.array(dist) < R)[0]] ) for R in R_bins])
        # --------------------------------------------------------

        # Let's save the data of these profiles
        gr.create_dataset('R_bins_sub', data = R_bins)
        gr.create_dataset('M_DM_sub', data = M)
        gr.create_dataset('M_stars_sub', data = M_stars)   
        if flag_gas:
            gr.create_dataset('M_gas_sub', data = M_gas)
        # --------------------------------------------------------

        # Let's compute the rotation matrix taking into accunt the inertia tensor
        rot_mat_IT, L_IT = compute_rot_mat_inertia(stars_c, stars_m, Rmax = 2 * properties[i, 13])
        # --------------------------------------------------------

        # Let's compute the rotation matrix taking into accunt the angular momentum tensor
        rot_mat_AM, L_AM = compute_rot_mat_angMom(stars_c, stars_v, stars_m, Rmax = 2 * properties[i, 13])
        # --------------------------------------------------------

        # Let's rotate the coordiantes with AM
        dm_c_rot_AM = dm_c @ rot_mat_AM
        dm_v_rot_AM = dm_v @ rot_mat_AM
        stars_c_rot_AM = stars_c @ rot_mat_AM
        stars_v_rot_AM = stars_v @ rot_mat_AM
        if flag_gas:
            gas_c_rot_AM = gas_c @ rot_mat_AM
            gas_v_rot_AM = gas_v @ rot_mat_AM

        L_AM_rot_AM = L_AM @ rot_mat_AM
        L_IT_rot_AM = L_IT @ rot_mat_AM
        # --------------------------------------------------------

        # Let's rotate the coordiantes with IT
        dm_c_rot_IT = dm_c @ rot_mat_IT
        dm_v_rotv = dm_v @ rot_mat_IT
        stars_c_rot_IT = stars_c @ rot_mat_IT
        stars_v_rot_IT = stars_v @ rot_mat_IT
        if flag_gas:
            gas_c_rot_IT = gas_c @ rot_mat_IT
            gas_v_rot_IT = gas_v @ rot_mat_IT

        L_AM_rot_IT = L_AM @ rot_mat_IT
        L_IT_rot_IT = L_IT @ rot_mat_IT
        # --------------------------------------------------------

        # Let's aligendthe stars with the IT
        x_stars_IT  = stars_c_rot_IT[:,0]
        y_stars_IT  = stars_c_rot_IT[:,1]
        z_stars_IT  = stars_c_rot_IT[:,2]
        vx_stars_IT = stars_v_rot_IT[:,0]
        vy_stars_IT = stars_v_rot_IT[:,1]
        vz_stars_IT = stars_v_rot_IT[:,2]
        # --------------------------------------------------------


        # Let's move to cylindrical coordinates and the kinematical properties
        r_stars_IT         = np.sqrt(x_stars_IT**2 + y_stars_IT**2)
        phi_stars_IT       = np.arctan2(y_stars_IT, x_stars_IT)
        jz_stars_IT        = x_stars_IT * vy_stars_IT - y_stars_IT * vx_stars_IT
        Erot_stars_IT      = stars_m * (jz_stars_IT**2) / (r_stars_IT**2)
        Ek_stars_IT        = stars_m * (vx_stars_IT**2 + vy_stars_IT**2 + vz_stars_IT**2)
        kappa_stars_IT     = np.sum(Erot_stars_IT) / np.sum(Ek_stars_IT)
        vphi_full_stars_IT = jz_stars_IT / r_stars_IT
        # --------------------------------------------------------


        # Now aligend the stars with the AM

        x_stars_AM  = stars_c_rot_AM[:,0]
        y_stars_AM  = stars_c_rot_AM[:,1]
        z_stars_AM  = stars_c_rot_AM[:,2]
        vx_stars_AM = stars_v_rot_AM[:,0]
        vy_stars_AM = stars_v_rot_AM[:,1]
        vz_stars_AM = stars_v_rot_AM[:,2]
        # --------------------------------------------------------

        # Let's move to cylindrical coordinates and the kinematical properties
        r_stars_AM     = np.sqrt(x_stars_AM**2 + y_stars_AM**2)
        phi_stars_AM   = np.arctan2(y_stars_AM, x_stars_AM)
        jz_stars_AM    = x_stars_AM * vy_stars_AM - y_stars_AM * vx_stars_AM
        Erot_stars_AM  = stars_m * (jz_stars_AM**2) / (r_stars_AM**2)
        Ek_stars_AM    = stars_m * (vx_stars_AM**2 + vy_stars_AM**2 + vz_stars_AM**2)
        kappa_stars_AM = np.sum(Erot_stars_AM) / np.sum(Ek_stars_AM)
        vphi_full_stars_AM = jz_stars_AM / r_stars_AM
        # --------------------------------------------------------

        # Let's save the main properties
        properties[i, 14] = np.dot(L_IT, L_AM) / ( np.linalg.norm(L_IT) * np.linalg.norm(L_AM) )
        properties[i, 15] = kappa_stars_AM
        properties[i, 16] = kappa_stars_IT
        # --------------------------------------------------------

        # Let's compute rotation curve with gas

        # Let's aligend the gas with the IT
        if flag_gas:
            x_gas_IT  = gas_c_rot_IT[:,0]
            y_gas_IT  = gas_c_rot_IT[:,1]
            z_gas_IT  = gas_c_rot_IT[:,2]
            vx_gas_IT = gas_v_rot_IT[:,0]
            vy_gas_IT = gas_v_rot_IT[:,1]
            vz_gas_IT = gas_v_rot_IT[:,2]
            # --------------------------------------------------------

            # Let's move to cylindrical coordinates and compute the kinematical properties
            r_gas_IT         = np.sqrt(x_gas_IT**2 + y_gas_IT**2)
            phi_gas_IT       = np.arctan2(y_gas_IT, x_gas_IT)
            jz_gas_IT        = x_gas_IT * vy_gas_IT - y_gas_IT * vx_gas_IT
            Erot_gas_IT      = gas_m * (jz_gas_IT**2) / (r_gas_IT**2)
            Ek_gas_IT        = gas_m * (vx_gas_IT**2 + vy_gas_IT**2 + vz_gas_IT**2)
            kappa_gas_IT     = np.sum(Erot_gas_IT) / np.sum(Ek_gas_IT)
            vphi_full_gas_IT = jz_gas_IT / r_gas_IT
            # --------------------------------------------------------

            # Now let's aligend the gas with the AM

            x_gas_AM  = gas_c_rot_AM[:,0]
            y_gas_AM  = gas_c_rot_AM[:,1]
            z_gas_AM  = gas_c_rot_AM[:,2]
            vx_gas_AM = gas_v_rot_AM[:,0]
            vy_gas_AM = gas_v_rot_AM[:,1]
            vz_gas_AM = gas_v_rot_AM[:,2]

            # Let's move to cylindrical coordinates and compute the kinematical properties
            r_gas_AM     = np.sqrt(x_gas_AM**2 + y_gas_AM**2)
            phi_gas_AM   = np.arctan2(y_gas_AM, x_gas_AM)
            jz_gas_AM    = x_gas_AM * vy_gas_AM - y_gas_AM * vx_gas_AM
            Erot_gas_AM  = gas_m * (jz_gas_AM**2) / (r_gas_AM**2)
            Ek_gas_AM    = gas_m * (vx_gas_AM**2 + vy_gas_AM**2 + vz_gas_AM**2)
            kappa_gas_AM = np.sum(Erot_gas_AM) / np.sum(Ek_gas_AM)
            vphi_full_gas_AM = jz_gas_AM / r_gas_AM
            # --------------------------------------------------------

            # Let's compute the binned rotational curves and saved it
            v_rot_gas_IT, bin_edges,_ = binned_statistic(r_gas_IT, np.abs(vphi_full_gas_IT), 'mean', bins = R_bins)
            v_std_gas_IT,_,_ = binned_statistic(r_gas_IT, np.abs(vphi_full_gas_IT), 'std', bins = R_bins)
            v_rot_gas_AM,_,_ = binned_statistic(r_gas_AM, np.abs(vphi_full_gas_AM), 'mean', bins = R_bins)
            v_std_gas_AM,_,_ = binned_statistic(r_gas_AM, np.abs(vphi_full_gas_IT), 'std', bins = R_bins)

        v_rot_stars_IT, bin_edges,_  = binned_statistic(r_stars_IT, np.abs(vphi_full_stars_IT), 'mean', bins = R_bins)
        v_std_stars_IT,_,_ = binned_statistic(r_stars_IT, np.abs(vphi_full_stars_IT), 'std', bins = R_bins)
        v_rot_stars_AM,_,_ = binned_statistic(r_stars_AM, np.abs(vphi_full_stars_AM), 'mean', bins = R_bins)
        v_std_stars_AM,_,_ = binned_statistic(r_stars_AM, np.abs(vphi_full_stars_AM), 'std', bins = R_bins)

        bin_width = (bin_edges[1] - bin_edges[0])
        bin_centers = bin_edges[1:] - bin_width/2

        gr.create_dataset('R_bins_vels', data = bin_centers)
        if flag_gas:
            gr.create_dataset('V_rot_gas_IT', data = v_rot_gas_IT)
            gr.create_dataset('V_std_gas_IT', data = v_std_gas_IT)
            gr.create_dataset('V_rot_gas_AM', data = v_rot_gas_AM)
            gr.create_dataset('V_std_gas_AM', data = v_std_gas_AM)
        gr.create_dataset('V_rot_stars_IT', data = v_rot_stars_IT)
        gr.create_dataset('V_std_stars_IT', data = v_std_stars_IT)
        gr.create_dataset('V_rot_stars_AM', data = v_rot_stars_AM)
        gr.create_dataset('V_std_stars_AM', data = v_std_stars_AM)
        # ---------------------------------------------------------------------------------------

        # Let's estimate the real profiles taking into account the halo
        print('Starting the estimation of properties with halo particles for galaxy ' + str(ids))

        # Let's load the DM particles of the halo to which the subhalo belongs
        #dm_halo = il.snapshot.loadHalo('/home/tnguser/sims.TNG/TNG100-1/output/', 99, gid, 'dm', fields=['Coordinates']) / h # If you have tng local files
        dm_halo = get('http://www.tng-project.org/api/TNG100-1/snapshots/99/halos/' + str(gid) + '/' + 'cutout.hdf5', {'dm':'Coordinates'})[0] / h
        # ---------------------------------------------------------------------------------------

        # If the halo is near the border let's center it
        dm_halo = np.where(dm_halo > 32500, dm_halo - 75000, dm_halo) 
        dm_halo = np.where(dm_halo < -32500, dm_halo + 75000, dm_halo)
        # ---------------------------------------------------------------------------------------

        # Now let's put the dm particle coordinates with respect to the center of the halo
        dm_halo = dm_halo - center_sub
        # ---------------------------------------------------------------------------------------

        # Compute the distance of each particle to the center and sum in radial bins
        dist = []
        for d in dm_halo:
            D = np.sqrt(sum([c**2 for c in d]))
            if D < 100: dist.append(D)
        R_bins = np.geomspace(1, 100, 20)
        M = np.array([len(np.where(np.array(dist) < R)[0]) * M_dm for R in R_bins])
        # ---------------------------------------------------------------------------------------

        # Let's load the stars particles of the halo to which the subhalo belongs
        #stars_halo = il.snapshot.loadHalo('/home/tnguser/sims.TNG/TNG100-1/output/', 99, gid, 'stars', fields=['Coordinates']) / h
        #masses = il.snapshot.loadHalo('/home/tnguser/sims.TNG/TNG100-1/output/', 99, gid, 'stars', fields=['Masses']) * 1e10 / h
        stars_halo, masses = get('http://www.tng-project.org/api/TNG100-1/snapshots/99/halos/' + str(gid) + '/' + 'cutout.hdf5', {'stars':'coordinates,masses'})
        stars_halo = stars_halo / h
        masses = masses * 1e10 / h
        # ---------------------------------------------------------------------------------------

        # If the halo is near the border let's center it
        stars_halo = np.where(stars_halo > 32500, stars_halo - 75000, stars_halo)
        stars_halo = np.where(stars_halo < -32500, stars_halo + 75000, stars_halo)
        # ---------------------------------------------------------------------------------------

        # Now let's put the stars particle coordinates with respect to the center of the halo
        stars_halo = stars_halo - center_sub
        # ---------------------------------------------------------------------------------------

        # Compute the distance of each particle to the center and sum in radial bins
        dist = []
        for d in stars_halo:
            D = np.sqrt(sum([c**2 for c in d]))
            if D < 100: dist.append(D)
        M_stars = np.array([np.sum( masses[np.where(np.array(dist) < R)[0]] ) for R in R_bins])
        # ---------------------------------------------------------------------------------------

        # Let's load the stars particles of the halo to which the subhalo belongs
        #gas_halo = il.snapshot.loadHalo('/home/tnguser/sims.TNG/TNG100-1/output/', 99, gid, 'gas', fields=['Coordinates']) / h
        #masses = il.snapshot.loadHalo('/home/tnguser/sims.TNG/TNG100-1/output/', 99, gid, 'gas', fields=['Masses']) * 1e10 / h
        gas_halo, masses = get('http://www.tng-project.org/api/TNG100-1/snapshots/99/halos/' + str(gid) + '/' + 'cutout.hdf5', {'gas':'coordinates,masses'})
        gas_halo = gas_halo / h
        masses   = masses * 1e10 / h
        # ---------------------------------------------------------------------------------------

        # If the halo is near the border let's center it
        gas_halo = np.where(gas_halo > 32500, gas_halo - 75000, gas_halo)
        gas_halo = np.where(gas_halo < -32500, gas_halo + 75000, gas_halo)
        # ---------------------------------------------------------------------------------------

        # Now let's put the stars particle coordinates with respect to the center of the halo
        gas_halo = gas_halo - center_sub
        # ---------------------------------------------------------------------------------------

        # Compute the distance of each particle to the center and sum in radial bins
        dist = []
        for d in gas_halo:
            D = np.sqrt(sum([c**2 for c in d]))
            if D < 100: dist.append(D)
        M_gas = np.array([np.sum( masses[np.where(np.array(dist) < R)[0]] ) for R in R_bins])

        gr.create_dataset('R_bins', data = R_bins)
        gr.create_dataset('M_DM', data = M)
        gr.create_dataset('M_stars', data = M_stars)   
        gr.create_dataset('M_gas', data = M_gas)
        # ---------------------------------------------------------------------------------------
        
        # Analysis flag:
        properties[i, 17] = 1
        # ---------------------------------------------------------------------------------------

        if (i % 10) == 0:
            print('Saving properties-----')
            # After 10 subhalos let's save the data and start again
            properties = np.delete(properties, np.where(properties[:,2] == 0)[0], axis = 0)

            if len(properties[:,0] > 0):
                if flag_MainProps:
                    properties = np.vstack((old_MainProps, properties))
                    del data['MainProps']
                    data.create_dataset('MainProps', data = properties)
                else:
                    data.create_dataset('MainProps', data = properties)
            data.close()
            data = h5py.File('../data/gals_properties.h5', 'a')
            try:
                flag_MainProps = True
                old_MainProps = data['MainProps'][()]
            except:
                flag_MainProps = False
            properties = np.zeros((nsubhalos, 18))
