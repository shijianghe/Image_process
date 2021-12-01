# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 10:00:01 2021

@author: v-olegkozlov
"""
import numpy as np
import scipy.ndimage

from checkerboard import detect_checkerboard

def find_bright(image, threshold = 0.2):
    image_high = np.max(image)
    image_low = np.min(image)
    
    threshold = image_low + (image_high - image_low) * threshold
    
    return np.where(image > threshold)


def get_corners(white_image):
    coord = find_bright(white_image)

    y_min = np.min(coord[0])
    y_max = np.max(coord[0])
    x_min = np.min(coord[1])
    x_max = np.max(coord[1])
    
    return x_min,y_min, x_max, y_max


def get_rotation_angle(white_image):
    top_left = white_image[:white_image.shape[0]//2,:white_image.shape[1]//2]

    coord = find_bright(top_left)
    
    y_top_left = np.min(coord[0])
    x_top_left = np.min(coord[1])
    
    top_rigth = white_image[:white_image.shape[0]//2,white_image.shape[1]//2:]

    coord = find_bright(top_rigth)
    
    y_top_right = np.min(coord[0])
    x_top_right = white_image.shape[1]//2 + np.max(coord[1])
    
    length = x_top_right - x_top_left
    delta = y_top_right - y_top_left
    
    return np.rad2deg(delta/length)



def crop_image(image, roi):
    x_min, y_min, x_max, y_max = roi
    return image[y_min:y_max,x_min:x_max]



def rotate_image(image, angle):
    rotated = scipy.ndimage.rotate(image, angle)
    return rotated



def sigmoid_enhancement (image, stretching_factor = 100, min_max_scaling = True):
    max_intensity = np.max(image)
    min_intensity = np.min(image)
    
    image = (image - min_intensity) / (max_intensity - min_intensity)
    image = image - 0.5
    image = image*stretching_factor
    
    converted_image = 1/(1 + np.exp(-image))
    
    if not min_max_scaling:
        converted_image = converted_image * (max_intensity - min_intensity)
        converted_image = converted_image + min_intensity
        
    return converted_image




def enhance_image(image, bin_h = 8, bin_v = 5, enhancement_function = sigmoid_enhancement,
                  window_overlap = 0.1):
    
    h = image.shape[0]
    v = image.shape[1]
    
    image_enhanced = np.zeros_like(image)
    
    h_start = 0
    h_end = 0
    
    v_start = 0
    v_end = 0
    
    while h_end<h:
        h_end = min(h_start + h//bin_h, h)
        v_start = 0
        v_end = 0
        
        while v_end<v:
            v_end = min(v_start + v//bin_v, v)
            image_slice = image[h_start:h_end, v_start:v_end]
            
            
        
            image_slice = enhancement_function(image_slice)
               
            image_enhanced[h_start:h_end, v_start:v_end] = image_slice
            v_start = int(v_end - (v_end - v_start) * window_overlap)
            
        h_start = int(h_end - (h_end - h_start) * window_overlap)
        
    return image_enhanced

def compensate_distortion(image, k_x = 0.03, k_y = 0.06):
    h = image.shape[0]
    w = image.shape[1]
    
    # meshgrid for interpolation mapping
    x,y = np.meshgrid(np.float32(np.arange(w)),np.float32(np.arange(h)))
    
    # center and scale the grid for radius calculation (distance from center of image)
    x_c = w/2 
    y_c = h/2 
    x = x - x_c
    y = y - y_c
    x = x/x_c
    y = y/y_c
    
    radius = np.sqrt(x**2 + y**2)
    
    # radial distortion model
    m_r_x = 1 + k_x*radius**2 
    m_r_y = 1 + k_y*radius**2
    
    # apply the model 
    x= x * m_r_x 
    y = y * m_r_y
    
    # reset all the shifting
    x= x*x_c + x_c
    y = y*y_c + y_c
    
    compensated = scipy.ndimage.map_coordinates(image, [y.ravel(),x.ravel()])
    
    compensated.resize(image.shape)
    
    return compensated

def bin_image (image, bining = 8):
    h,w = image.shape
    
    image_binned = image[:h//bining*bining, :w//bining*bining].\
    reshape(h//bining, bining, w//bining, bining).mean(-1).mean(1)  
    
    return image_binned

def find_checkerboard(check_image, rows = 5, cols =8):
    corners, score = detect_checkerboard(check_image, (cols,rows))
    
    if score>0.1:
        print("Checkerboard not found! Trying enhanced image")
        check_image_enhanced = enhance_image(check_image)
        corners, score = detect_checkerboard(check_image_enhanced, (cols,rows))
        if score>0.1:
            raise Exception("Checkerboard not found!")
    
    return corners.astype(np.int32)

def side_adjustment (squares, area):
    side_coefficient = area ** 0.5
    squares_corrected = np.zeros_like(squares)
    
    for i in range(squares.shape[0]):
        for j in range(squares.shape[1]):
            coordinates = squares[i,j]
            x_shift = (coordinates[2] - coordinates[0]) * (1-side_coefficient) / 2
            y_shift = (coordinates[3] - coordinates[1]) * (1-side_coefficient) / 2
            
            squares_corrected[i,j,0] = round(coordinates[0] + x_shift)
            squares_corrected[i,j,1] = round(coordinates[1] + y_shift)
            squares_corrected[i,j,2] = round(coordinates[2] - x_shift)
            squares_corrected[i,j,3] = round(coordinates[3] - y_shift)
            
    return squares_corrected.astype(np.int32)

def get_check_rois(check_image, rows = 5, cols =8, area = 0.75):
    """
    return squares : np.array of (rows+1, cols+1, 4 size)
        Each 4-element array represents top-left, bottom-right coordinates
        of the respective ckecker ROI in a format:
        (x_tl, y_tl, x_br, y_br)

    """
    corners = find_checkerboard(check_image, rows, cols)
    
    rows_edges = rows+2
    cols_edges = cols+2
    coordinates = np.zeros((rows_edges,cols_edges,2))
    
    x = np.array([i[0][0] for i in corners])
    y = np.array([i[0][1] for i in corners])
    
    x_matrix = np.reshape(x, (cols,rows)).T
    y_matrix = np.reshape(y, (cols,rows)).T
    
    coordinates[1:-1, 1:-1, 0] = x_matrix
    coordinates[1:-1, 1:-1, 1] = y_matrix
    
    coordinates[0,1:-1,0] = coordinates[1,1:-1,0]
    coordinates[-1,1:-1,0] = coordinates[-2,1:-1,0]
    coordinates[:,-1,0] = check_image.shape[1]
    
    coordinates[ 1:-1 ,0, 1] = coordinates[1:-1,1,1]
    coordinates[1:-1, -1, 1] = coordinates[1:-1,-2,1]
    coordinates[-1,:,1] = check_image.shape[0]
    
    squares = np.zeros((rows+1,cols+1,4))
    for i in range(squares.shape[0]):
        for j in range(squares.shape[1]):
            squares[i,j] = np.array([coordinates[i,j,0],
                                     coordinates[i,j,1],
                                     coordinates[i+1,j+1,0], 
                                     coordinates[i+1,j+1,1]]).astype(np.int32)
      
    squares = side_adjustment(squares, area)    
    
    return squares

def get_intensities(image, squares):
    intensities = np.zeros((squares.shape[0],squares.shape[1]))
    for i in range(squares.shape[0]):
        for j in range(squares.shape[1]):
            intensities[i,j] = get_mean(image, squares[i,j])
    return intensities

def get_mean(image,roi):
    return np.mean(image[roi[1]:roi[3], roi[0]:roi[2]])

def calculate_local_contrast(intensities, indices):
    neigboring_intensities = np.zeros(4)
    neigboring_intensities[:] = np.nan
    
    index_pairs = [[0,1], [0,-1], [1,0], [-1,0]]
    
    for i, index_pair in enumerate(index_pairs):
        neighbor_i = indices[0] + index_pair[0]
        neighbor_j = indices[1] + index_pair[1]

        if neighbor_i>=0 and neighbor_i<intensities.shape[0] and neighbor_j >=0 and neighbor_j < intensities.shape[1]:
            neigboring_intensities[i] = intensities[neighbor_i,neighbor_j]
    
    neighbors_average = np.nanmean(neigboring_intensities)
    return neighbors_average/intensities[indices[0],indices[1]]

def get_local_check_contrast(intensities, pattern):
    contrast = np.zeros((intensities.shape[0],intensities.shape[1]))
    contrast[:,:] = np.nan
    for i in range(intensities.shape[0]):
        for j in range(intensities.shape[1]):
            if pattern[i,j]:
                contrast[i,j] = calculate_local_contrast(intensities, [i,j])
    return contrast

def generate_pattern(rows, cols, check_type = "+"):
    """
    check_type : TYPE, optional
        DESCRIPTION. The default is "+".

    Returns
    -------
    np-array of (rows,cols) with True in cells for which contrast needs
    to be calculated

    """
    pattern = np.indices((rows,cols)).sum(axis=0) % 2
    pattern = pattern.astype(bool)
    if check_type == 'w':
        pattern[:,:] = True
    elif check_type == "-":
        pattern = np.logical_not(pattern)
    return pattern
    
def get_roi_from_angles(image, center, angle_roi=(3,3), full_FOV=(40,30), padding = (5,5)):
    """
    return ROI coordinates in pixels : 
        (x_tl, y_tl, x_br, y_br)

    """
    
    H_size= image.shape[1]
    V_size = image.shape[0]
    pixel_per_deg_H = H_size/full_FOV[0]
    pixel_per_deg_V = V_size/full_FOV[1]
    
    half_size_H = angle_roi[0]*pixel_per_deg_H/2
    half_size_V = angle_roi[1]*pixel_per_deg_V/2
    
    center_position_H = (H_size/2) + center[0]*pixel_per_deg_H
    center_position_V = (V_size/2) + center[1]*pixel_per_deg_V
    
    x_tl = max(center_position_H - half_size_H, padding[0])
    y_tl = max(center_position_V - half_size_V, padding[1])
    x_br = min(center_position_H + half_size_H, H_size-padding[0])
    y_br = min(center_position_V + half_size_V, V_size-padding[1])
    
    return int(x_tl), int(y_tl), int(x_br), int(y_br)

def calculate_avg_in_roi(image, coord):
    x_start = coord[0]
    x_end = coord[2]
    y_start = coord[1]
    y_end = coord[3]
    
    image_slice = image[y_start:y_end, x_start: x_end]
    return np.mean(image_slice)