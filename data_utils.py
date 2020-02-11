#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 20:26:35 2018

@author: subhayanmukherjee
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi


def get_normalized_slc_amp_by_tanhmz(img):
   points = img
   a = np.abs(points)
   mad = np.median(np.abs(a - np.median(a)))
   mz = 0.6745*((a-np.median(a))/mad)
   mz = (np.tanh(mz/3.5) + 1 )/2
   return mz


def is_outlier(points, thresh=3.5):
    """    
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = 0
    diff = np.abs(points - median)
    med_abs_deviation = np.median(diff)
    modified_z_score = 0.6745 * (points - median)/ med_abs_deviation
    return modified_z_score > thresh
def saturate_outlier(img):
    A = np.sqrt( np.absolute(img) )
    angle = np.angle(img) 
    s = A.shape
    x = is_outlier( A.reshape(-1,1) )
    mask = 1- x.reshape(s)
    A1 = mask*A 
    A1 /= np.max(A1)    
    A1[np.logical_and(mask==0,A>np.median(A) )] = 1
    return A1*np.exp(1j*angle)

#usage
#Z_processed = saturate_outlier(z)



def apply_hysteresis_threshold(image, low, high):
    """Apply hysteresis thresholding to `image`.
    This algorithm finds regions where `image` is greater than `high`
    OR `image` is greater than `low` *and* that region is connected to
    a region greater than `high`.
    Parameters
    ----------
    image : array, shape (M,[ N, ..., P])
        Grayscale input image.
    low : float, or array of same shape as `image`
        Lower threshold.
    high : float, or array of same shape as `image`
        Higher threshold.
    Returns
    -------
    thresholded : array of bool, same shape as `image`
        Array in which `True` indicates the locations where `image`
        was above the hysteresis threshold.
    Examples
    --------
    >>> image = np.array([1, 2, 3, 2, 1, 2, 1, 3, 2])
    >>> apply_hysteresis_threshold(image, 1.5, 2.5).astype(int)
    array([0, 1, 1, 1, 0, 0, 0, 1, 1])
    References
    ----------
    .. [1] J. Canny. A computational approach to edge detection.
           IEEE Transactions on Pattern Analysis and Machine Intelligence.
           1986; vol. 8, pp.679-698.
           DOI: 10.1109/TPAMI.1986.4767851
    """
    
    low = np.clip(low, a_min=None, a_max=high)  # ensure low always below high
    mask_low = image > low
    mask_high = image > high
    # Connected components of mask_low
    labels_low, num_labels = ndi.label(mask_low)
    # Check which connected components contain pixels from mask_high
    sums = ndi.sum(mask_high, labels_low, np.arange(num_labels + 1))
    connected_to_high = sums > 0
    thresholded = connected_to_high[labels_low]
    
    return thresholded



def imshow(img, title='',big=0, cmap='jet',save='-1',fsize = 15):    
    
    if len(img.shape) == 4:
        img = img[0,:,:,0]
    if not big:
        
        if str( type(img[0][0]) )[14:17] == 'com':            
            print('Complex type')
            if save is not '-1':#Saves angle ony                
                
                plt.figure(figsize=(fsize,fsize))
                plt.imshow(np.angle(img), cmap = cmap)
                plt.colorbar(fraction = 0.046, pad= 0.04)
                plt.title(title)
                plt.savefig( str(save)+"_phase.png" )
                
                plt.figure(figsize=(fsize,fsize))
                plt.title(title)
                plt.imshow(np.log(np.absolute(img)), cmap = cmap)#                
                plt.colorbar(fraction = 0.046, pad= 0.04)
                plt.savefig( str(save)+"_magnitude.png" )
            else:
                
                plt.subplot(1,2,1)
                plt.imshow(np.absolute(img), cmap = cmap)
                plt.title(title+" amplitude")
                plt.colorbar(fraction = 0.046, pad= 0.04)            
                plt.subplot(1,2,2)
                plt.imshow(np.angle(img), cmap = cmap)
                plt.title(title+" angle")            
                plt.tight_layout()
        else:            
            if save is not '-1':
                
                plt.figure(figsize=(fsize,fsize))
                plt.imshow(img, cmap = cmap)
                plt.title(title)
                plt.colorbar(fraction = 0.046, pad= 0.04)
                plt.savefig( str(save)+".png" )
                
            else:
                
                plt.imshow(img, cmap = cmap)
                plt.title(title)
                
    else:
        
        plt.figure(figsize=(fsize,fsize))
        plt.imshow(np.absolute(img), cmap = cmap)
        plt.title(title)
        plt.colorbar(fraction = 0.046, pad= 0.04)
#        plt.figure(figsize=(fsize,fsize))
#        plt.imshow(np.angle(img), cmap = cmap)
#        plt.title(title+" angle")
    
    
    plt.colorbar(fraction = 0.046, pad= 0.04)
    plt.show()