import os
import cv2
import logging
import numpy as np
import torch
from medpy import metric


logging.basicConfig(filename='eval.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

gt_dir = r'E:\Experiment\data\TestDataset'
results_dir = r'result_path'

model_name = 'model_name'
logging.info(model_name)
mDice = 0
mJaccard = 0
SDdice = 0

def _dice_loss(score, target):
    target = target.astype(float)
    smooth = 1
    intersect = np.sum(score * target)
    y_sum = np.sum(target * target)
    z_sum = np.sum(score * score)
    dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    return dice

for dataset in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:

    gt_root = gt_dir + '\\{}\\masks'.format(dataset)
    results_root = results_dir + '\\{}'.format(dataset)
    gt_file = os.listdir(gt_root)
    results_file = os.listdir(results_root)

    n = len(gt_file)

    for gt_name, result_name in zip(gt_file, results_file):
        if gt_name.endswith('.png') and result_name.endswith('.png'):
            gt_path = os.path.join(gt_root, gt_name)
            result_path = os.path.join(results_root, result_name)

            gt_image = cv2.imread(gt_path, 0)
            result_image = cv2.imread(result_path, 0)
           
            gt_image = (gt_image - gt_image.min()) / (gt_image.max() - gt_image.min() + 1e-8)
            result_image = (result_image - result_image.min()) / (result_image.max() - result_image.min() + 1e-8)
            gt_image[gt_image > 0] = 1
            result_image[result_image > 0] = 1
           
            dice = metric.binary.dc(result_image, gt_image)
            jaccard = metric.binary.jc(result_image, gt_image)
            sddice = _dice_loss(result_image, gt_image)

            mDice += dice
            mJaccard += jaccard
            SDdice += sddice

    mDice /= n
    mJaccard /= n
    SDdice /= n
  
    logging.info('dataset,{}, mDice: {:0.4f}, mJaccard: {:0.4f}, SDdice: {:0.4f}'
                 .format(dataset, mDice, mJaccard, SDdice))

