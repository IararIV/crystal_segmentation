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


def rename_folder(path, dataset_name, type_name):
    assert str(type_name) in ["recon", "gt"]
    filenames = glob.glob(path + "*.tif")
    print("DATASET:", dataset_name, "TYPE:", type_name, "N_FILES:", len(filenames))
    for n, f in enumerate(filenames):
        new_name = str(dataset_name) + "_" + str(type_name) + "_" + str(n).zfill(5) + ".tif"
        os.rename(f, path + new_name)
        
def pad_tensors(tensor, shape):
    h, w = tensor.shape
    nh, nw = shape
    up_down = (int(np.floor((nh - h)/2)), int(np.ceil((nh - h)/2)))
    left_right = (int(np.floor((nw - w)/2)), int(np.ceil((nw - w)/2)))
    tensor = np.pad(tensor, (up_down, left_right), 'constant')
    return tensor

def normalise_im(im):
    return (im - im.min())/(im.max() - im.min())

def pad_folder(path, shape, normalise=True):
    filenames = glob.glob(path)
    for f in filenames:
        im = Image.open(f)
        im = np.array(im)
        if normalise:
            im = normalise_im(im)
        im = pad_tensors(im, shape)
        im = Image.fromarray(im)
        im.save(f)
        
def get_args():
    parser = argparse.ArgumentParser(description='Create the folder with the training data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--dir_img', dest='dir_img', type=str, default='/dls/tmp/lqg38422/TRAIN/recon/',
                        help='Path to the folder containing the images')
    parser.add_argument('-m', '--dir_mask', dest='dir_mask', type=str, default='/dls/tmp/lqg38422/TRAIN/gt/',
                        help='Path to the folder containing the masks')
    parser.add_argument('-f', '--dir_data', dest='dir_data', type=str, default='/dls/science/users/lqg38422/DATA/',
                        help='Path to the folder containing the datasets')
    parser.add_argument('-d', '--datasets', dest='datasets', nargs='+', default=["13068", "13076", "13246", "13270", "13295", "13551", "13724", "13737", "13769", "14253"],
                        help='List of dataset names') #test_datasets = ["13284"]
    parser.add_argument('-p', '--padding', dest='padding', default=(900, 900),
                        help='Shape to pad too')
    
    return parser.parse_args()
        
        
if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    
    logging.info(f'Data:\n'
                 f'\tDatasets directory : {args.dir_data}\n'
                 f'\tImage output directory : {args.dir_img}\n'
                 f'\tMask output directory : {args.dir_mask}\n'
                 f'\tDatasets : {args.datasets}\n'
                 f'\tPadding shape : {args.padding}\n')
    
    train_datasets = args.datasets
    train_path_reco = args.dir_img
    train_path_gt = args.dir_mask
    shape = args.padding
 
    # Add training data
    t = time.time()
    for dataset in train_datasets:
        print(f"--- DATASET {dataset} ---")
        
        if multi_axis == True:
            pass
        
        else:
            recon_path = args.dir_data + str(dataset) + "/recon/"
            gt_path = args.dir_data + str(dataset) + "/gt/"

            # Copy files
            print("Copying recon...")
            for filename in glob.glob(recon_path + "*"):
                shutil.copy(filename, train_path_reco)
            print("Copying gt...")
            for filename in glob.glob(gt_path + "*"):
                shutil.copy(filename, train_path_gt)
            print("Done!")

    # Pad all files
    print("Padding recon...")
    pad_folder(train_path_reco + "*", shape, normalise=True)
    print("Done!")
    print("Padding gt...")
    pad_folder(train_path_gt + "*", shape, normalise=False)
    print("Done!")
    
    res = time.time() - t
    print("Time:", res)