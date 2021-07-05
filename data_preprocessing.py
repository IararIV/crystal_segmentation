import argparse
import logging
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import glob
import shutil
import time

              
def get_args():
    parser = argparse.ArgumentParser(description='Create the folder with the training data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--dir_img', dest='dir_img', type=str, default='/dls/tmp/lqg38422/TRAIN/recon/',
                        help='Path to the folder where images will be stored')
    parser.add_argument('-m', '--dir_mask', dest='dir_mask', type=str, default='/dls/tmp/lqg38422/TRAIN/gt/',
                        help='Path to the folder where masks will be stored')
    parser.add_argument('-f', '--dir_data', dest='dir_data', type=str, default='/dls/science/users/lqg38422/DATA/',
                        help='Path to the folder containing the datasets')
    parser.add_argument('-a', '--multi_axis', dest='multi_axis', type=bool, default=True,
                        help='Bool indicating if 3 axis will be stored')
    parser.add_argument('-d', '--datasets', dest='datasets', nargs='+', default=["13068", "13076", "13246", "13270", "13295", "13551", "13724", "13737", "13769", "14253"],
                        help='List of dataset names') #test_datasets = ["13284"]
    
    return parser.parse_args()
        
        
if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    
    logging.info(f'Data:\n'
                 f'\tDatasets directory : {args.dir_data}\n'
                 f'\tImage output directory : {args.dir_img}\n'
                 f'\tMask output directory : {args.dir_mask}\n'
                 f'\tMultiple axis : {args.multi_axis}\n'
                 f'\tDatasets : {args.datasets}\n')
    
    train_datasets = args.datasets
    train_path_reco = args.dir_img
    train_path_gt = args.dir_mask
 
    # Add training data
    t = time.time()
    for dataset in train_datasets:
        print(f"--- DATASET {dataset} ---")
        
        recon_path = args.dir_data + str(dataset) + "/recon/"
        gt_path = args.dir_data + str(dataset) + "/gt/"
        
        if args.multi_axis == True:
            # Load the tensors (reconstruction and ground truth) into memory
            print("Loading tensors into memory...")
            sample = []
            segment = []
            for filename in glob.glob(recon_path + "*"):
                sample.append(np.array(Image.open(filename)))
            sample = np.array(sample)
            if sample.dtype == np.float32:
                sample = np.uint16((sample - sample.min()) / (sample.max() - sample.min()) * 65535)
            for filename in glob.glob(gt_path + "*"):
                im = np.array(Image.open(filename))
                segment.append(im)
            segment = np.array(segment)
            print("Done!")
            # Save for each axis
            print("XY axis:")
            for xy in range(len(sample[:,0,0])):
                # Recon
                filename = dataset + "XY_recon_" + str(xy).zfill(5) + ".tif"
                im = Image.fromarray(sample[xy,:,:])
                im.save(train_path_reco + filename)
                # Ground truth
                filename = dataset + "XY_gt_" + str(xy).zfill(5) + ".tif"
                im = Image.fromarray(segment[xy,:,:])
                im.save(train_path_gt + filename)
            print("Done!")
            print("XZ axis:")
            for xz in range(len(sample[0,:,0])):
                # Recon
                filename = dataset + "XZ_recon_" + str(xz).zfill(5) + ".tif"
                im = Image.fromarray(sample[:,xz,:])
                im.save(train_path_reco + filename)
                # Ground truth
                filename = dataset + "XZ_gt_" + str(xz).zfill(5) + ".tif"
                im = Image.fromarray(segment[:,xz,:])
                im.save(train_path_gt + filename)
            print("Done!")
            print("YZ axis:")
            for yz in range(len(sample[0,0,:])):
                # Recon
                filename = dataset + "YZ_recon_" + str(yz).zfill(5) + ".tif"   
                im = Image.fromarray(sample[:,:,yz])
                im.save(train_path_reco + filename)
                # Ground truth
                filename = dataset + "YZ_gt_" + str(yz).zfill(5) + ".tif"   
                im = Image.fromarray(segment[:,:,yz])
                im.save(train_path_gt + filename)
            print("Done!")
        
        else:
            # Copy files
            print("Copying recon...")
            for filename in glob.glob(recon_path + "*"):
                shutil.copy(filename, train_path_reco)
            print("Copying gt...")
            for filename in glob.glob(gt_path + "*"):
                shutil.copy(filename, train_path_gt)
            print("Done!")
    
    res = time.time() - t
    print("Time:", res)