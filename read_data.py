import os
import cv2
import matplotlib.pyplot as plt
import skimage.io
import numpy as np
from numpy import array
import random
from random import randint
from sklearn.preprocessing import normalize


def read(path, file_count):
    window = 28 * 8
    
    filename_list = os.listdir(path)
    filename_list = [f for f in filename_list if f.endswith(".tiff")]
    filename_list = filename_list[:file_count]
    
    data_size = file_count * 49
    
    Xtrain = np.zeros((data_size, window, window, 3), dtype=np.float32)
    Ytrain = np.zeros((data_size, window, window), dtype=np.float32)
    
    index=0
    for filename in filename_list:
        
        image_path = os.path.join(path, filename)
        image1 = skimage.io.imread(image_path,plugin='tifffile') / 255
        
        gt_filename = os.path.splitext(filename)[0]
        gt_path = os.path.join(path, "gt", gt_filename + ".tif")
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise Exception(f"Ground truth file not found: {gt_path}")

        gt[gt>0] = 1
        
        height, width = image1.shape[:2]
         
        stepx = int(width / window) + 1
        stepy = int(height / window) + 1
        
        for i in range (stepx):
            for j in range(stepy):
                
                coorx = i * window
                coory = j * window
                
                if coorx + window > width:
                    coorx = width - window 
                if coory + window > height:
                    coory = height - window
                
            
                image_patch = image1[coory:coory+window,coorx:coorx+window]
                image_label = gt[coory:coory+window,coorx:coorx+window]
        
                Ytrain[index,:,:] = image_label.copy()
                Xtrain[index,:,:,:] = image_patch.copy()
             
                index +=1
    
    y_train = Ytrain.reshape((len(Ytrain), window, window, 1))    
    x_train = Xtrain
            
    print(f"Loaded {len(x_train)} patches")
    print(f"Image patch shape: {x_train[0].shape}, Mask patch shape: {y_train[0].shape}")

    return x_train, y_train


