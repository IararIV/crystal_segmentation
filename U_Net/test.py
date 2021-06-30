import argparse
import logging
import os
import sys

import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt

# python test.py -p /dls/tmp/lqg38422/TEST/gt/ -m /dls/tmp/lqg38422/TEST/gt/

def get_args():
    parser = argparse.ArgumentParser(description='Get metrics to evaluate the predictions of the U-Net',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--dir_pred', dest='dir_pred', type=str, default='/dls/tmp/lqg38422/PREDS/',
                        help='Path to the folder containing the images (/path/to/preds/)')
    parser.add_argument('-m', '--dir_mask', dest='dir_mask', type=str, default='/dls/tmp/lqg38422/TEST/gt/',
                        help='Path to the folder containing the masks (/path/to/masks/)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    
    preds_files = glob.glob(args.dir_pred + "*")
    masks_files = glob.glob(args.dir_mask + "*")
    
    background_error = []
    crystal_error = []
    loop_error = []
    liquor_error = []
    
    for n in range(len(preds_files)):
        pred = np.array(Image.open(preds_files[n]))
        mask = np.array(Image.open(masks_files[n]))
                
        error0 = np.sum(pred[pred == mask] == 0) / np.sum(mask == 0) * 100
        error1 = np.sum(pred[pred == mask] == 1) / np.sum(mask == 1) * 100
        error2 = np.sum(pred[pred == mask] == 2) / np.sum(mask == 2) * 100
        error3 = np.sum(pred[pred == mask] == 3) / np.sum(mask == 3) * 100
        
        background_error.append(error0)
        crystal_error.append(error1)
        loop_error.append(error2)
        liquor_error.append(error3)
        
    plt.figure()
    plt.axis([None,None,0,100])
    plt.title("Test error - Pixels correctly predicted / Total pixels (for each class))")
    plt.plot(background_error)
    plt.plot(crystal_error)
    plt.plot(loop_error)
    plt.plot(liquor_error)
    plt.legend(["Background error", "Crystal error", "Loop error", "Liquor error"])
    plt.show()      
