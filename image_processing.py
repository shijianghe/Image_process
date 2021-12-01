# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 10:23:09 2021

@author: v-olegkozlov
"""

import image_tools
import numpy as np

def pre_process_image(image, angle, corners,corners_2, k_x, k_y, bin_size, padding = (0,0)):
    binned = image_tools.bin_image(image, bin_size)
    min_value = np.min(binned)
    binned_rotated = image_tools.rotate_image(binned, angle)
    binned_rotated_cropped = image_tools.crop_image(binned_rotated,corners)
    binned_rotated_cropped_compensated = image_tools.compensate_distortion(binned_rotated_cropped, k_x = 0.04, k_y = 0.06)
    cropped_2 = image_tools.crop_image(binned_rotated_cropped_compensated,corners_2)
    cropped_2[cropped_2==0] = min_value
    processed = cropped_2[padding[1]:cropped_2.shape[0]-padding[1],
                          padding[0]:cropped_2.shape[1]-padding[0]]
    return processed

def get_parameters_from_white(white_image, bin_size, k_x,k_y):
    angle = image_tools.get_rotation_angle(white_image)
    white_binned = image_tools.bin_image(white_image,bin_size)
    white_binned_rotated = image_tools.rotate_image(white_binned, angle)
    corners = image_tools.get_corners(white_binned_rotated)
    white_binned_rotated_cropped = image_tools.crop_image(white_binned_rotated,corners)
    white_binned_rotated_cropped_compensated = image_tools.compensate_distortion(white_binned_rotated_cropped, k_x = 0.04, k_y = 0.06)
    corners_2 = image_tools.get_corners(white_binned_rotated_cropped_compensated)
    
    return angle, corners, corners_2

    

def calculate_local_checkerboard_contrast(intensities_plus, intensities_minus):
    rows = intensities_plus.shape[0]
    cols = intensities_plus.shape[1]
    
    pattern_plus = image_tools.generate_pattern(rows,cols, '+')
    contrast_plus = image_tools.get_local_check_contrast(intensities_plus, pattern_plus)
   
    pattern_minus = image_tools.generate_pattern(rows,cols, '-')
    contrast_minus = image_tools.get_local_check_contrast(intensities_minus, pattern_minus)  
    
    contrast_combined = np.nan_to_num(contrast_plus, False) + np.nan_to_num(contrast_minus, False) 
    
    return contrast_combined


def calculate_sequential_checkerboard_contrast(intensities_plus, intensities_minus):
    
    seq_check_contrast = intensities_plus/intensities_minus
    seq_check_contrast[seq_check_contrast<1] = 1/seq_check_contrast[seq_check_contrast<1]
    
    return seq_check_contrast

def calculate_sequential_WB_contrast(white, black):
    return white/black

def calculate_uniformity(image):
    return image/np.max(image) * 100

def calculate_statistics(array):
    min_value = np.nanmin(array)
    max_value = np.nanmax(array)
    average_value = np.nanmean(array)
    SD = np.nanstd(array)
    return min_value, max_value, average_value, SD

def calculate_ansi(image, ansi_array, edge_array, angle_roi = (3,3), full_FOV = (40,30)):
    outer_coords = [0,2,4,10,12,14,20,22,24] #linear coordinates for edges
    
    
    ansi_roi_array = []
    for coordinates in ansi_array:
        current_roi = image_tools.get_roi_from_angles(image, coordinates, angle_roi, full_FOV)
        ansi_roi_array.append(current_roi)
    
    edge_roi_array = []
    for coordinates in edge_array:
        current_roi = image_tools.get_roi_from_angles(image, coordinates, angle_roi, full_FOV)
        edge_roi_array.append(current_roi)
        
    ansi_avg_array = []
    edges_avg_array = []
    
    for i in range(len(ansi_array)):
        current_avg = image_tools.calculate_avg_in_roi(image, ansi_roi_array[i])
        ansi_avg_array.append(current_avg)
        
    for i in range(len(edge_array)):
       current_avg = image_tools.calculate_avg_in_roi(image, edge_roi_array[i])
       edges_avg_array.append(current_avg)
       
    ansi_avg_matrix = np.reshape(ansi_avg_array, (3,3))
    ansi_uniformity = np.empty(25)
    ansi_uniformity[:] = np.nan
    
    for i, index in enumerate(outer_coords):
        ansi_uniformity[index] = edges_avg_array[i]
    
    ansi_uniformity = ansi_uniformity.reshape((5,5))
    ansi_uniformity[1:4,1:4] = ansi_avg_matrix
    ansi_uniformity = ansi_uniformity/ansi_uniformity[2,2]
    
    return ansi_uniformity, ansi_roi_array, edge_roi_array