import argparse
import logging
import os
import sys

import numpy as np
from PIL import Image
import timeit

import tomophantom
from tomophantom import TomoP3D
from tomophantom.supp.qualitymetrics import QualityTools
from tomophantom.supp.flatsgen import synth_flats
from tomobar.supp.suppTools import normaliser

def normalise_im(im):
    return (im - im.min())/(im.max() - im.min())

def create_sample(dataset, model, N_size, angles_num, output_path_recon, output_path_gt):
    
    print ("Building 3D phantom using TomoPhantom software")
    tic=timeit.default_timer()
    model = model # select a model number from the library
    N_size = N_size #512 # Define phantom dimensions using a scalar value (cubic phantom)
    path = os.path.dirname(tomophantom.__file__)
    path_library3D = os.path.join(path, "Phantom3DLibrary.dat")
    #This will generate a N_size x N_size x N_size phantom (3D)
    phantom_tm = TomoP3D.Model(model, N_size, path_library3D)
    toc=timeit.default_timer()
    Run_time = toc - tic
    print("Phantom has been built in {} seconds".format(Run_time))
    
    # Get groundtruth
    values = np.unique(phantom_tm)
    assert len(values) == 4, "More than 4 classes where generated, run again"
    GROUND_TRUTH = np.zeros(phantom_tm.shape)
    GROUND_TRUTH[phantom_tm == values[0]] = 0
    GROUND_TRUTH[phantom_tm == values[1]] = 3
    GROUND_TRUTH[phantom_tm == values[2]] = 1
    GROUND_TRUTH[phantom_tm == values[3]] = 2

    # Projection geometry related parameters:
    Horiz_det = N_size # int(np.sqrt(2)*N_size) # detector column count (horizontal)
    Vert_det = N_size # detector row count (vertical) (no reason for it to be > N)
    angles_num = angles_num # int(0.5*np.pi*N_size); # angles number
    angles = np.linspace(0.0,179.9,angles_num,dtype='float32') # in degrees
    angles_rad = angles*(np.pi/180.0)

    print ("Building 3D analytical projection data with TomoPhantom")
    projData3D_analyt = TomoP3D.ModelSino(model, N_size, Horiz_det, Vert_det, angles, path_library3D)

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

    print ("Normalise projections using ToMoBAR software")
    # normalise the data, the required data format is [detectorsX, Projections, detectorsY]
    projData3D_norm = normaliser(projData3D_noisy, flatsSIM, darks=None, log='true', method='mean')

    for n in range(projData3D_norm.shape[1]):
        # save image
        filename = output_path_recon + str(dataset) + "_recon_" + str(n).zfill(5) + ".tif"
        im = projData3D_norm[:,n,:].astype(np.float64)
        im = normalise_im(im)
        im = Image.fromarray(im)
        im.save(filename)
        
    for n in range(GROUND_TRUTH.shape[1]):
        # save image
        filename = output_path_gt + str(dataset) + "_gt_" + str(n).zfill(5) + ".tif"
        im = GROUND_TRUTH[:,n,:]
        im = Image.fromarray(im.astype(np.uint8))
        im.save(filename)
        
def get_args():
    parser = argparse.ArgumentParser(description='Create the folder with the training data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--images', metavar='IMAGES', dest='dir_img',
                        help='Folder where the ouput images will be stored')
    parser.add_argument( '-m', '--masks', metavar='MASKS', dest='dir_mask',
                        help='Folder where the ouput masks will be stored')
    parser.add_argument('-n', '--n_datasets', dest='n_datasets', 
                        help='Number of datasets to be generated')
    parser.add_argument('-s', '--size', dest='size', default=900,
                        help='Size of the data generated')
    parser.add_argument('-a', '--angles', dest='angles', default=800,
                        help='Number of angles')
    
    return parser.parse_args()



if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    
    logging.info(f'Simulating data:\n'
                 f'\tImage output directory : {args.dir_img}\n'
                 f'\tMask output directory : {args.dir_mask}\n'
                 f'\tNumber of datasets : {args.n_datasets}\n'
                 f'\tSize : {args.size}\n'
                 f'\tAngles : {args.angles}\n')
    
    dataset = "00000"
    model = 17
    N_datasets = int(args.n_datasets)
    N_size = int(args.size)
    angles_num = int(args.angles)
    output_path_recon = args.dir_img
    output_path_gt = args.dir_mask
    
    for i in range(N_datasets):
        print("Creating dataset", dataset)
        create_sample(dataset, model, N_size, angles_num, output_path_recon, output_path_gt)
        dataset = str(i+1).zfill(5)