import argparse
import logging
import os
import sys

import numpy as np
from PIL import Image
import glob

# python test.py -p /dls/tmp/lqg38422/TEST/gt/ -m /dls/tmp/lqg38422/TEST/gt/

def MSE(preds, targets):
    mse = ((preds - targets)**2).mean(axis=None)
    return mse

def IOU(result1, result2):
    intersection = np.logical_and(result1, result2)
    union = np.logical_or(result1, result2)
    tmp = np.sum(intersection, axis=None) / np.sum(union, axis=None)
    iou_score = tmp if not np.isnan(tmp) else 0
    return iou_score

def dice_coef(preds, targets, smooth=1):
    intersection = np.sum(preds * targets)
    union = np.sum(preds) + np.sum(targets)
    dice = (2. * intersection + smooth)/(union + smooth)
    dice = dice if not np.isnan(dice) else 0
    return dice

def get_args():
    parser = argparse.ArgumentParser(description='Get metrics to evaluate the predictions of the U-Net',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--dir_pred', dest='dir_pred', type=str, default='/dls/tmp/lqg38422/PREDS/',
                        help='Path to the folder containing the images')
    parser.add_argument('-m', '--dir_mask', dest='dir_mask', type=str, default='/dls/tmp/lqg38422/TEST/gt/',
                        help='Path to the folder cantaining the masks')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    
    preds_files = glob.glob(args.dir_pred + "*")
    masks_files = glob.glob(args.dir_mask + "*")
    
    MSE_metric = 0
    IOU_metric = 0
    DC_metric = 0
    
    for n in range(len(preds_files)):
        pred = np.array(Image.open(preds_files[n]))
        mask = np.array(Image.open(masks_files[n]))
        
        MSE_metric += MSE(pred, mask)
        IOU_metric += IOU(pred, mask)
        DC_metric += dice_coef(pred, mask)
        
    MSE_metric /= len(preds_files)
    IOU_metric /= len(preds_files)
    DC_metric /= len(preds_files)
    
    message = f"TEST:\nMSE: {MSE_metric}\nIOU: {IOU_metric}\nF1 Score: {DC_metric}\n"
    
    print(message)
    
    
    
    
