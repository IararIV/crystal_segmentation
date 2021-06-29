#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import timeit
import os
import matplotlib.pyplot as plt
import numpy as np
import tomophantom
from tomophantom import TomoP3D
from tomophantom.supp.qualitymetrics import QualityTools
from tomophantom.supp.flatsgen import synth_flats
from tomobar.supp.suppTools import normaliser

print ("Building 3D phantom using TomoPhantom software")
tic=timeit.default_timer()
model = 17 # select a model number from the library
N_size = 256 # Define phantom dimensions using a scalar value (cubic phantom)
path = os.path.dirname(tomophantom.__file__)
path_library3D = os.path.join(path, "Phantom3DLibrary.dat")
#This will generate a N_size x N_size x N_size phantom (3D)
phantom_tm = TomoP3D.Model(model, N_size, path_library3D)
toc=timeit.default_timer()
Run_time = toc - tic
print("Phantom has been built in {} seconds".format(Run_time))

sliceSel = int(0.5*N_size)
#plt.gray()
plt.figure() 
plt.subplot(131)
plt.imshow(phantom_tm[sliceSel,:,:],vmin=0, vmax=1)
plt.title('3D Phantom, axial view')

plt.subplot(132)
plt.imshow(phantom_tm[:,sliceSel,:],vmin=0, vmax=1)
plt.title('3D Phantom, coronal view')

plt.subplot(133)
plt.imshow(phantom_tm[:,:,sliceSel],vmin=0, vmax=1)
plt.title('3D Phantom, sagittal view')
plt.show()

# Projection geometry related parameters:
Horiz_det = int(np.sqrt(2)*N_size) # detector column count (horizontal)
Vert_det = N_size # detector row count (vertical) (no reason for it to be > N)
angles_num = int(0.5*np.pi*N_size); # angles number
angles = np.linspace(0.0,179.9,angles_num,dtype='float32') # in degrees
angles_rad = angles*(np.pi/180.0)

#%%
print ("Building 3D analytical projection data with TomoPhantom")
projData3D_analyt= TomoP3D.ModelSino(model, N_size, Horiz_det, Vert_det, angles, path_library3D)

intens_max_clean = np.max(projData3D_analyt)
sliceSel = 150
plt.figure() 
plt.subplot(131)
plt.imshow(projData3D_analyt[:,sliceSel,:],vmin=0, vmax=intens_max_clean)
plt.title('2D Projection (analytical)')
plt.subplot(132)
plt.imshow(projData3D_analyt[sliceSel,:,:],vmin=0, vmax=intens_max_clean)
plt.title('Sinogram view')
plt.subplot(133)
plt.imshow(projData3D_analyt[:,:,sliceSel],vmin=0, vmax=intens_max_clean)
plt.title('Tangentogram view')
plt.show()

#%%
print ("Simulate synthetic flat fields, add flat field background to the projections and add noise")
I0  = 15000; # Source intensity
flatsnum = 100 # the number of the flat fields simulated

[projData3D_noisy, flatsSIM] = synth_flats(projData3D_analyt,
                                           source_intensity = I0, source_variation=0.02,\
                                           arguments_Bessel = (1,10,10,12),\
                                           specklesize = 5,\
                                           kbar = 0.3,\
                                           jitter = 1.0,\
                                           sigmasmooth = 3, flatsnum=flatsnum)
plt.figure() 
plt.subplot(121)
plt.imshow(projData3D_noisy[:,0,:])
plt.title('2D Projection (before normalisation)')
plt.subplot(122)
plt.imshow(flatsSIM[:,0,:])
plt.title('A selected simulated flat-field')
plt.show()

#%%
print ("Normalise projections using ToMoBAR software")
# normalise the data, the required data format is [detectorsX, Projections, detectorsY]
projData3D_norm = normaliser(projData3D_noisy, flatsSIM, darks=None, log='true', method='mean')

intens_max = 0.15*np.max(projData3D_norm)
sliceSel = 150
plt.figure() 
plt.subplot(131)
plt.imshow(projData3D_norm[:,sliceSel,:],vmin=0, vmax=intens_max)
plt.title('Normalised 2D Projection (erroneous)')
plt.subplot(132)
plt.imshow(projData3D_norm[sliceSel,:,:],vmin=0, vmax=intens_max)
plt.title('Sinogram view')
plt.subplot(133)
plt.imshow(projData3D_norm[:,:,sliceSel],vmin=0, vmax=intens_max)
plt.title('Tangentogram view')
plt.show()

#%%
print ("Normalise projections dynamically using ToMoBAR software")
# normalise the data, the required data format is [detectorsX, Projections, detectorsY]
projData3D_norm_dyn = normaliser(projData3D_noisy, flatsSIM, darks=None, log='true',  method='dynamic', dyn_downsample=2, dyn_iterations=10)

intens_max = 0.15*np.max(projData3D_norm_dyn)
sliceSel = 150
plt.figure() 
plt.subplot(131)
plt.imshow(projData3D_norm_dyn[:,sliceSel,:],vmin=0, vmax=intens_max)
plt.title('Normalised dynamically 2D Projection (erroneous)')
plt.subplot(132)
plt.imshow(projData3D_norm_dyn[sliceSel,:,:],vmin=0, vmax=intens_max)
plt.title('Sinogram view')
plt.subplot(133)
plt.imshow(projData3D_norm_dyn[:,:,sliceSel],vmin=0, vmax=intens_max)
plt.title('Tangentogram view')
plt.show()