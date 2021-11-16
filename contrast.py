'''

Author: Shijiang He, shih@microsoft.com


'''
#TODO ANSI uniformity
#TODO normalize by center not by max

#import cv2
#from PIL import Image 

import numpy as np

#import itertools, math, os
import matplotlib.pyplot as plt

from checkerboard import detect_checkerboard 
from RIC import RIC_DB
from intersect import intersection

import os

import seaborn as sns


channels = ['R', 'G', 'B', 'W']
modes = ['+chkbrd', '-chkbrd', 'white', 'black']
suffixex = ["50mA"]

edges = [[-18,13.5],
         [0,13.5],
         [18, 13.5],
         [-18,0],
         [0,0],
         [18,0],
         [-18,-13.5],
         [0,-13.5],
         [18,-13.5]
    ]

ANSI = [[-13.33,10],
         [0,10],
         [13.33, 10],
         [-13.33,0],
         [0,0],
         [13.33,0],
         [-13.33,-10],
         [0,-10],
         [13.33,-10]
    ]


def do(database_path, Rows=6, Cols=9, Bin = 8):
    
    fig_height = 10
    
    fig_cols = ['+Check Raw', '-Check Raw', 'Check Local', 'Check Seq', 
                'White Uniformity', 'Black Uniformity', 'W/B Seq', 'ANSI Raw', 'ANSI uniformity']#, 'White Raw Local',
#                'White Local','Black Raw Local', 'Black Local']

    fig_rows = ['R','G','B','W']

    
    fig, axs = plt.subplots(len(fig_rows), len(fig_cols))
    fig.set_figheight(fig_height)
    fig.set_figwidth(fig_height * len(fig_cols)/len(fig_rows) * 1.5)
    
    save_file = database_path.split('.')[0] + "/"
    
    DMA_name = save_file.split('/')[-2]
    
    if not os.path.exists(save_file):
        os.makedirs(save_file)
    
    try:
        Cols = int(Cols)
        Rows = int(Rows)
    except:
        print('Rows and Cols are numbers')
        raise

    db = RIC_DB(database_path)
    measurements_in_db = db.get_meas_list()

    print('Image database name: ', database_path)
    print('Measurements in this DB:')
    print(measurements_in_db)
    
    stats = {}
    
    px = py = 0
    
    PD_for_sequential = [0, 0]
    
    
    
    for k, suffix in enumerate(suffixex):
        for i, channel in enumerate(channels):
            for j, mode in enumerate(modes):
                
                
                measurement = f"{channel}_{mode}_{suffix}"
                print(f"\nProcessing measurement {measurement}\n")
        
                try:
                    MeasurementID = db.index_by_desc(measurement)
                except:
                    print(f"Measurement {measurement} not in DB")
                    continue
                
                image = db.read_luminance(MeasurementID)
                image = bin_image(image, Bin)
                
                
                if '+chkbrd' in measurement:
                    
                    try:
                        px, py = get_vertices(image, Rows, Cols)
                        PD_for_sequential[0] = get_means(image, px, py, Rows, Cols) 
                        contrast_array_plus = get_local(image, Rows, Cols,  px, py, axs[i, 0])
                    except:
                        print(f"Error processing {measurement}!\n")
                        continue

                    
                    
                    
                elif '-chkbrd' in measurement:
                    
                    try:
                        px, py = get_vertices(image, Rows, Cols)
                        PD_for_sequential[1] = get_means(image, px, py, Rows, Cols) 
                        contrast_array_minus = get_local(image, Rows, Cols,  px, py, axs[i, 1])
                    except:
                        print(f"Error processing {measurement}!\n")
                        continue
                    

                    contrast_combined = np.nan_to_num(contrast_array_plus, False) + \
                    np.nan_to_num(contrast_array_minus, False) 
                    
                    stats['checkerboard local'] = {'min' : np.min(contrast_combined),
                                                   'max' : np.max(contrast_combined),
                                                   'avg' : np.mean(contrast_combined),
                                                   'SD' : np.std(contrast_combined),
                                                   }
                                        

                    ax = axs[i, 2]
                    ax.imshow(contrast_combined)
                    contour = ax.imshow(contrast_combined)
                    plt.colorbar(contour, ax=ax)
                    
                    ax.set_xlabel(f"""min: {stats['checkerboard local']['min']:.1f} max: {stats['checkerboard local']['max']:.1f} avg: {stats['checkerboard local']['avg']:.1f} SD {stats['checkerboard local']['SD']:.1f}""")
                    
                    
                    current_file = save_file+measurement+"_local.csv"
                    np.savetxt(current_file, contrast_combined, delimiter=',')
                    
                    seq_check_contrast = PD_for_sequential[1]/PD_for_sequential[0]
                    
                    PD_whites = PD_whites = np.concatenate(\
                        (PD_for_sequential[0][::2], PD_for_sequential[1][1::2]))
                    
                    PD_blacks = np.concatenate(\
                        (PD_for_sequential[0][1::2], PD_for_sequential[1][::2]))
                     
                    seq_check_avg2 = np.mean(PD_whites)/np.mean(PD_blacks)
                     
                    mask = np.where(seq_check_contrast < 1)[0]
                    seq_check_contrast[mask] = 1/seq_check_contrast[mask]
                    
                    seq_check_contrast = seq_check_contrast.reshape((Rows, Cols))
                    
                    stats['checkerboard seq'] = {'min' : np.min(seq_check_contrast),
                               'max' : np.max(seq_check_contrast),
                               'avg' : np.mean(seq_check_contrast),
                               'SD' : np.std(seq_check_contrast),
                               'avg2' : seq_check_avg2
                               }
                    

                    ax = axs[i, 3]                    
                    ax.imshow(seq_check_contrast)
                    contour = ax.imshow(seq_check_contrast)
                    plt.colorbar(contour, ax=ax)                   
                    ax.set_xlabel(f"""min: {stats['checkerboard seq']['min']:.1f} max: {stats['checkerboard seq']['max']:.1f} avg: {stats['checkerboard seq']['avg']:.1f} SD: {stats['checkerboard seq']['SD']:.1f} avg2: {stats['checkerboard seq']['avg2']:.1f}""")
                   
                    
                    current_file = save_file+measurement+"_sequential.csv"            
                    np.savetxt(current_file, seq_check_contrast, delimiter=',')
                    
                    
                    
                    
                    
                elif 'white' in measurement:
# =============================================================================
#                     contrast_array = get_local_from_uniform(image, Rows, Cols,  px, py, axs[i,7])
#                     
#                     stats['white local'] = {'min' : np.min(contrast_array),
#                                                 'max' : np.max(contrast_array),
#                                                 'avg' : np.mean(contrast_array),
#                                                 'SD' : np.std(contrast_array),
#                                                 }
# 
#                     
#                     ax = axs[i, 8]
#                     ax.imshow(contrast_combined)
#                     contour = ax.imshow(contrast_array)
#                     plt.colorbar(contour, ax=ax)
#                     ax.set_xlabel(f"""min: {stats['white local']['min']:.1f} max: {stats['white local']['max']:.1f} avg: {stats['white local']['avg']:.1f} SD {stats['white local']['SD']:.1f}""")
#                    
#                    
#                     current_file = save_file+measurement+"_local.csv"
#                     np.savetxt(current_file, contrast_array, delimiter=',')
#                        
# =============================================================================
                    coord = np.where (image>image.mean())
                    
                    y_min = np.min(coord[0])
                    y_max = np.max(coord[0])
                    x_min = np.min(coord[1])
                    x_max = np.max(coord[1])
                    
                    white_img = image[y_min:y_max,x_min:x_max]
                    

                    
                    
                    
                    coord_y = np.where (white_img[:, white_img.shape[1]//2]>image.mean())
                    y_min_rect = np.min(coord_y[0])+1
                    y_max_rect = np.max(coord_y[0])-1
                    
                    coord_x = np.where (white_img[white_img.shape[0]//2, :]>image.mean())
                    x_min_rect = np.min(coord_x[0])+1
                    x_max_rect = np.max(coord_x[0])-1
                    
                    ansi_uniformity, ansi_roi_array, edges_roi_array = \
                    calculate_ansi(white_img, [x_min_rect,x_max_rect,y_min_rect, y_max_rect])
                    
                    white_center = white_img[y_min_rect:y_max_rect,x_min_rect:x_max_rect]
                    white_av = white_center.mean()
                    
                    white_uniformity = white_img/np.max(white_img) * 100
                    
                    white_uniformity_area = white_uniformity[y_min_rect:y_max_rect,x_min_rect:x_max_rect]
                    
                    stats['white uniformity'] = {'min' : np.min(white_uniformity_area),
                             'max' : np.max(white_uniformity_area),
                             'avg' : np.mean(white_uniformity_area),
                             'SD' : np.std(white_uniformity_area),
                             }
                    
                    
                    
                    ax = axs[i, 4]
                    contour = ax.imshow(white_uniformity)
                    ax.plot([x_min_rect, x_min_rect, x_max_rect, x_max_rect, x_min_rect], 
                            [y_min_rect,y_max_rect,y_max_rect,y_min_rect, y_min_rect], 
                            color = 'r', linewidth=0.5)
                    
                    ax.set_xlabel(f"""min: {stats['white uniformity']['min']:.1f} max: {stats['white uniformity']['max']:.1f} avg: {stats['white uniformity']['avg']:.1f} SD {stats['white uniformity']['SD']:.1f}""")
 
                    
                    plt.colorbar(contour, ax=ax)   
                    
                    current_file = save_file+measurement+"_unoiformity.csv"
                    np.savetxt(current_file, white_uniformity_area, delimiter=',')  
                    
                    
                                    
                    stats['ANSI uniformity'] = {'min' : np.nanmin(ansi_uniformity),
                             'max' : np.nanmax(ansi_uniformity),
                             'avg' : np.nanmean(ansi_uniformity),
                             'SD' : np.nanstd(ansi_uniformity),
                             }
                
                    
                    ax = axs[i,7]
                    contour = ax.imshow(white_img)
                    plt.colorbar(contour, ax = ax)
                    for coord in ansi_roi_array:
                        ax.plot([coord[i % 4][0] for i in range(5)], 
                            [coord[i % 4][1] for i in range(5)], 
                            color = 'r', linewidth=1)

                    for coord in edges_roi_array:
                        ax.plot([coord[i % 4][0] for i in range(5)], 
                            [coord[i % 4][1] for i in range(5)], 
                            color = 'b', linewidth=1)
                        
                        
                    ax = axs[i,8]
                    sns.heatmap(ansi_uniformity, annot = True, fmt = ".2f", 
                                xticklabels=False, yticklabels=False, cmap='viridis', ax = ax)
                
                    ax.set_xlabel(f"""min: {stats['ANSI uniformity']['min']:.1f} max: {stats['ANSI uniformity']['max']:.1f} avg: {stats['ANSI uniformity']['avg']:.1f} SD {stats['ANSI uniformity']['SD']:.1f}""")

                    current_file = save_file+measurement+"_ANSI_uniformity.csv"
                    np.savetxt(current_file, ansi_uniformity, delimiter=',')  
                
                    
                    
                elif 'black' in measurement:
# =============================================================================
#                     contrast_array = get_local_from_uniform(image, Rows, Cols,  px, py, axs[i,9])
#                     
#                     stats['black local'] = {'min' : np.min(contrast_array),
#                                                 'max' : np.max(contrast_array),
#                                                 'avg' : np.mean(contrast_array),
#                                                 'SD' : np.std(contrast_array),
#                                                 }   
#                     
#                     ax = axs[i, 10]
#                     ax.imshow(contrast_combined)
#                     contour = ax.imshow(contrast_array)
#                     plt.colorbar(contour, ax=ax)                    
#                     ax.set_xlabel(f"""min: {stats['black local']['min']:.1f} max: {stats['black local']['max']:.1f} avg: {stats['black local']['avg']:.1f} SD {stats['black local']['SD']:.1f}""")
#                     
#                     
#                     current_file = save_file+measurement+"_local.csv"
#                     np.savetxt(current_file, contrast_array, delimiter=',')
#                     
# =============================================================================



                    black_img = image[y_min:y_max,x_min:x_max]
                    
                    black_center = black_img[y_min_rect:y_max_rect,x_min_rect:x_max_rect]

                    black_av = black_center.mean()
                    
                    
                    black_uniformity = black_img/np.max(black_img) * 100
                    
                    black_uniformity_area = black_uniformity[y_min_rect:y_max_rect,x_min_rect:x_max_rect]
                    
                    stats['black uniformity'] = {'min' : np.min(black_uniformity_area),
                    		 'max' : np.max(black_uniformity_area),
                    		 'avg' : np.mean(black_uniformity_area),
                    		 'SD' : np.std(black_uniformity_area),
                    		 }
           
                 

                        
                    ax = axs[i, 5]
                    contour = ax.imshow(black_uniformity)
                    
                    ax.plot([x_min_rect, x_min_rect, x_max_rect, x_max_rect, x_min_rect], 
                    		[y_min_rect,y_max_rect,y_max_rect,y_min_rect, y_min_rect], 
                    		color = 'r', linewidth=0.5)
                    
                    plt.colorbar(contour, ax=ax)   
                    ax.set_xlabel(f"""min: {stats['black uniformity']['min']:.1f} max: {stats['black uniformity']['max']:.1f} avg: {stats['black uniformity']['avg']:.1f} SD {stats['black uniformity']['SD']:.1f}""")
                    
                    
                    
                    sequential_contrast = white_av/black_av
                    
                    sequential_contrast_map = white_center/black_center
                    
                    
                    stats['B/W seq'] = {'min' : np.min(sequential_contrast_map),
                             'max' : np.max(sequential_contrast_map),
                             'avg' : np.mean(sequential_contrast_map),
                             'SD' : np.std(sequential_contrast_map),
                             }   

                    
                    stats['sequential average'] = {'avg' : sequential_contrast}
                    
                                       
                    ax = axs[i, 6]
                    contour = ax.imshow(sequential_contrast_map)
                    plt.colorbar(contour, ax=ax)   
                    ax.set_xlabel(f"""min: {stats['B/W seq']['min']:.1f} max: {stats['B/W seq']['max']:.1f} avg: {stats['B/W seq']['avg']:.1f} SD {stats['B/W seq']['SD']:.1f}""")
                    
                    
                    
                    current_file = save_file+measurement+"_stats.txt"
                    
                    with open(current_file, 'w') as f:
                        for key in stats:
                            f.write(key)
                            f.write('\n')
                            for key2, value2 in stats[key].items():
                                f.write(key2 + '\t' + str(value2) + '\n')
                
                else:
                    print (f"{measurement} not processed\n")

    # serial_no = os.path.splitext( os.path.split(database_path)[-1] )[0]
    axs[0,0].set_ylabel('R')
    axs[1,0].set_ylabel('G')    
    axs[2,0].set_ylabel('B')
    axs[3,0].set_ylabel('W')
    
    
    for i, title in enumerate(fig_cols):
        axs[0,i].set_title(title)

#    axs[0,7].set_title('White Raw Local')
#    axs[0,8].set_title('White Local')
#    axs[0,9].set_title('Black Raw Local')
#    axs[0,10].set_title('Black Local')
    

    fig.suptitle(DMA_name, size=16)
    plt.tight_layout()
    plt.savefig(save_file+f'plot_{DMA_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Done!\n")

def calculate_ansi(white_image, inner_coords):
    
    outer_coords = [0,2,4,10,12,14,20,22,24]
    
    corner_indices = [0,2,6,8]
    
    
    angle = 3
    FOV_H = 40
    FOV_V = 30
    FOV = [FOV_H, FOV_V]
    coeff_H = angle/FOV_H
    coeff_V = angle/FOV_V
    
    total_lenght_outer = [int(white_image.shape[1]), 
                       int(white_image.shape[0])]
    
    half_size_outer = [int(total_lenght_outer[0]*coeff_H/2), 
                       int(total_lenght_outer[1]*coeff_V/2)]
    
    total_lenght_inner = [inner_coords[1]-inner_coords[0],
                          inner_coords[3]-inner_coords[2]]
    

    
    half_size_inner = [int(total_lenght_inner[0]*coeff_H/2),
                      int(total_lenght_inner[1]*coeff_V/2)]
    
    #half_size_inner = half_size_outer
    
    shift = [int((total_lenght_outer[i] - total_lenght_inner[i])/2) for i in (0,1)]
    
    ansi_roi_array = []
    for i, coordinates in enumerate(ANSI):
        if i in corner_indices:
            current_roi =calculate_roi(coordinates, total_lenght_outer,half_size_inner, FOV, [0,0])
        else:
            current_roi =calculate_roi(coordinates, total_lenght_inner, half_size_outer, FOV, shift)
        
        ansi_roi_array.append(current_roi)
        
    edges_roi_array = []
    for i, coordinates in enumerate(edges):
        if i in corner_indices:
            current_roi =calculate_roi(coordinates, total_lenght_outer, half_size_inner, FOV, [0,0], [3,7])
        else:
            current_roi =calculate_roi(coordinates, total_lenght_inner,half_size_outer, FOV, shift, [2,2])
        
        edges_roi_array.append(current_roi)        
    
    
    ansi_avg_array = []
    edges_avg_array = []
    
    for i in range(len(ANSI)):
        current_avg = calculate_avg_in_roi(white_image, ansi_roi_array[i])
        ansi_avg_array.append(current_avg)
        
    for i in range(len(edges)):
       current_avg = calculate_avg_in_roi(white_image, edges_roi_array[i])
       edges_avg_array.append(current_avg)   
   
    ansi_avg_matrix = np.reshape(ansi_avg_array, (3,3))
    
    resulted_array = np.empty(25)
    resulted_array[:] = np.nan
    
    for i, index in enumerate(outer_coords):
        resulted_array[index] = edges_avg_array[i]
        
    resulted_array = resulted_array.reshape((5,5))
    resulted_array[1:4,1:4] = ansi_avg_matrix
    resulted_array = resulted_array/resulted_array[2,2]
        
    
    return resulted_array, ansi_roi_array, edges_roi_array

def calculate_avg_in_roi(white_image, coord):
    x_start = coord[0][0]
    x_end = coord[3][0]
    y_start = coord[0][1]
    y_end = coord[1][1]
    
    image_slice = white_image[y_start:y_end, x_start: x_end]
    return np.mean(image_slice)

def calculate_roi(center_angle,total_length, half_size, FOV, shift, padding = False):
    #corner arrangement:
    #0____3
    #|    |
    #1____2
    
    center_pix = [int(total_length[0]/2 * (1 + 2 * center_angle[0]/FOV[0]) + shift[0]),
                  int(total_length[1]/2 * (1 - 2 * center_angle[1]/FOV[1]) + shift[1])]
    coord = [0, 0, 0, 0]
    coord[0] = [int(center_pix[0] - half_size[0]), 
                int(center_pix[1] - half_size[1])]
                
    coord[1] = [int(center_pix[0] - half_size[0]), 
            int(center_pix[1] + half_size[1])]
                
    coord[2] = [int(center_pix[0] + half_size[0]), 
        int(center_pix[1] + half_size[1])]
                
    coord[3] = [int(center_pix[0] + half_size[0]), 
        int(center_pix[1] - half_size[1])]
    
    if padding and not center_angle == [0, 0]:
        if center_angle[0] > 0:
            for i in range(4):
                coord[i][0] = coord[i][0] - padding[0]
        elif center_angle[0]<0:
            for i in range(4):
                coord[i][0] = coord[i][0] + padding[0]
        
        if center_angle[1] < 0:
            for i in range(4):
                coord[i][1] = coord[i][1] - padding[1]
        elif center_angle[1]>0:
            for i in range(4):
                coord[i][1] = coord[i][1] + padding[1]    
                
    return coord            
                
    
def get_vertices(image, Rows, Cols):
    rows = Rows-1; ROWS=Rows+1
    cols = Cols-1; COLS=Cols+1

    # image = cv2.imread(file_name,cv2.IMREAD_GRAYSCALE)
    # image = np.asarray(image.convert('L')).astype(np.uint16)


    corners, score = detect_checkerboard(image, (cols,rows))

    serial_no = 'Test001'
    if score >0.9:
        print( f'Checkerboard is not found in {serial_no}')
        return False 

    x=corners[:,0,0].reshape((cols,rows)).transpose()
    y=corners[:,0,1].reshape((cols,rows)).transpose()

    X = np.full((ROWS, COLS), np.nan)
    Y = np.full((ROWS, COLS), np.nan)
    X[1:-1,1:-1] = x
    Y[1:-1,1:-1] = y

    Y[ 0,1:-1]= 2*y[ 0,:]-y[ 1,:]
    Y[-1,1:-1]= 2*y[-1,:]-y[-2,:]
    X[1:-1, 0]= 2*x[ :,0]-x[ :,1]
    X[1:-1,-1]= 2*x[:,-1]-x[:,-2]



    for i in range(1,Rows):
        f =  np.polyfit(X[i,1:-1], Y[i,1:-1], 2)
        #print( X[i,1:-1], Y[i,1:-1], 2)
        Y[i, 0] = np.polyval(f, X[i, 0])
        Y[i,-1] = np.polyval(f, X[i,-1])

    for i in range(1,Cols):
        f =  np.polyfit(Y[1:-1,i], X[1:-1, i], 2)
        #print( X[1:-1,i], Y[1:-1, i])
        X[ 0, i] = np.polyval(f, Y[ 0, i])
        X[-1, i] = np.polyval(f, Y[-1, i])


    row_pixels, col_pixels=image.shape
    xx = np.arange(0,col_pixels)
    yy = np.arange(0,row_pixels)
    
    x1,y1=X[ 0,1:-1], Y[ 0,1:-1]
    x3,y3=X[-1,1:-1], Y[-1,1:-1]
    x2,y2=X[1:-1, 0], Y[1:-1, 0]
    x4,y4=X[1:-1,-1], Y[1:-1,-1]
    
    #print(x3,y3,x4,y4)
    
    f1=np.polyfit(x1,y1,2); 
    f2=np.polyfit(y2,x2,2)
    f3=np.polyfit(x3,y3,2); 
    f4=np.polyfit(y4,x4,2)
    
        
    corner1 = intersection(xx, np.polyval(f1,xx), np.polyval(f2,yy), yy)
    corner2 = intersection(np.polyval(f2,yy),yy,  xx, np.polyval(f3,xx))
    corner3 = intersection(xx, np.polyval(f3,xx), np.polyval(f4,yy), yy)
    corner4 = intersection(np.polyval(f4,yy),yy,  xx, np.polyval(f1,xx))
    
    

    X[ 0, 0] = corner1[0][0]; Y[ 0, 0]=corner1[1][0]
    X[-1, 0] = corner2[0][0]; Y[-1, 0]=corner2[1][0]
    X[-1,-1] = corner3[0][0]; Y[-1,-1]=corner3[1][0]
    X[ 0,-1] = corner4[0][0]; Y[ 0,-1]=corner4[1][0]
    
    
    px = np.round(X).astype(np.int32)
    py = np.round(Y).astype(np.int32)
    
    return px, py


def bin_image(image, Bin):
    
    h,w = image.shape 
    image = image[:h//Bin*Bin, :w//Bin*Bin].reshape(h//Bin, Bin, w//Bin, Bin).mean(-1).mean(1)  
    return image
    
"""
def get_contrast(db, Rows, Cols, MeasurementID, Bin=8):

    coverage = 0.25

    image = db.read_luminance(MeasurementID)

    h,w = image.shape 

    image = image[:h//Bin*Bin, :w//Bin*Bin].reshape(h//Bin, Bin, w//Bin, Bin).mean(-1).mean(1)    




    rows = Rows-1; ROWS=Rows+1
    cols = Cols-1; COLS=Cols+1

    # image = cv2.imread(file_name,cv2.IMREAD_GRAYSCALE)
    # image = np.asarray(image.convert('L')).astype(np.uint16)


    corners, score = detect_checkerboard(image, (cols,rows))

    serial_no = 'Test001'
    if score >0.9:
        print( f'Checkerboard is not found in {serial_no}')
        return False 


    x=corners[:,0,1].reshape((cols,rows)).transpose()
    y=corners[:,0,0].reshape((cols,rows)).transpose()

    X = np.full((ROWS, COLS), np.nan)
    Y = np.full((ROWS, COLS), np.nan)
    X[1:-1,1:-1] = x
    Y[1:-1,1:-1] = y

    # Y[ 0,1:-1]=y[ 0,:]; X[ 0,1:-1]= 2*x[ 0,:]-x[ 1,:]
    # Y[-1,1:-1]=y[-1,:]; X[-1,1:-1]= 2*x[-1,:]-x[-2,:]
    # X[1:-1, 0]=x[:, 0]; Y[1:-1, 0]= 2*y[ :,0]-y[ :,1]
    # X[1:-1,-1]=x[:,-1]; Y[1:-1,-1]= 2*y[:,-1]-y[:,-2]

    # Y[ 0, 0] = Y[ 1, 0];  X[ 0, 0]=X[ 0, 1]
    # Y[-1, 0] = Y[-2, 0];  X[-1, 0]=X[-1, 1]
    # Y[-1,-1] = Y[-2,-1];  X[-1,-1]=X[-1,-2]
    # Y[ 0,-1] = Y[ 1,-1];  X[ 0,-1]=X[ 0,-2]


    X[ 0,1:-1]= 2*x[ 0,:]-x[ 1,:]
    X[-1,1:-1]= 2*x[-1,:]-x[-2,:]
    Y[1:-1, 0]= 2*y[ :,0]-y[ :,1]
    Y[1:-1,-1]= 2*y[:,-1]-y[:,-2]

    for i in range(1,Rows):
        f=np.polyfit(X[0,i])   


    # Y[ 0, 0] = Y[ 1, 0];  X[ 0, 0]=X[ 0, 1]
    # Y[-1, 0] = Y[-2, 0];  X[-1, 0]=X[-1, 1]
    # Y[-1,-1] = Y[-2,-1];  X[-1,-1]=X[-1,-2]
    # Y[ 0,-1] = Y[ 1,-1];  X[ 0,-1]=X[ 0,-2]




    fig = plt.figure()

    ax1=fig.add_subplot(211)
    contour=ax1.imshow(image, cmap='gray')
    plt.colorbar(contour, ax=ax1)


    px = X.astype(np.int32)
    py = Y.astype(np.int32)
   
    Squares=[]
    
    pitch_x = (X[-1,:].mean()-X[0,:].mean())/Rows
    pitch_y = (Y[:,-1].mean()-Y[:,0].mean())/Cols
    
    del_x = int(pitch_x*(1-np.sqrt(coverage))/2)
    del_y = int(pitch_y*(1-np.sqrt(coverage))/2)
    
    for i in range(Rows):
        squares=[]
        for j in range(Cols):
            x0, x1 = px[i,j]+del_x, px[i+1,j+1]-del_x
            y0, y1 = py[i,j]+del_y, py[i+1,j+1]-del_y
            
            squares.append( image[x0:x1, y0:y1].mean())
            
            ax1.plot([Y[i,j], Y[i+1,j], Y[i+1,j+1], Y[i,j+1], Y[i,j]],
                     [X[i,j], X[i+1,j], X[i+1,j+1], X[i,j+1], X[i,j]],
                    'r-'
                   )
            ax1.plot([y0,y1, y1, y0, y0], [x0, x0, x1, x1, x0],'g-')
        Squares.append(squares)
        
    PD_data=np.array(Squares).ravel()
    


    pattern = np.indices((Rows,Cols)).sum(axis=0) % 2
    pattern = pattern.astype(bool).ravel()

    sum_squares1 = PD_data[0::2].sum()
    sum_squares2 = PD_data[1::2].sum()

    # +/- checkerboard
    if sum_squares1<sum_squares2:
        pattern = np.logical_not(pattern)

    
    adj_squares_list = [ ((i,j+1), (i,j-1),(i-1,j),(i+1,j)) for i in range(Rows) for j in range(Cols)]
    adj_squares = [ list(filter(lambda xy: 0<=xy[0]<Rows and 0<=xy[1]<Cols, pos)) for pos in adj_squares_list] 
    adj_squares = [ np.array(squares).transpose() for squares in adj_squares]

    squares1d =[sq[0]*Cols+sq[1] for sq in adj_squares]




    contrast_array = np.full(Rows*Cols, np.nan)


    for i in range(Rows*Cols):
        if pattern[i]:
            contrast_array[i] =  PD_data[squares1d[i]].mean()/PD_data[i] 
    contrast_array = contrast_array.reshape((Rows, Cols))

    ax2=fig.add_subplot(212)
    contour = ax2.imshow(contrast_array)
    plt.colorbar(contour, ax=ax2)





    # styles=itertools.product(   ['-', '--', ':'] ,['+', 'o', '*', ',', 'v', 'd', 'x'], 'rgbkycm')

    # pos = np.arange(-H_roi, H_roi+1)
    # ax3=plt.subplot(223)

    # for x, y, xtalk, row, col, style in zip(X, Y, xtalks, Rows, Cols, styles ):
    #     data = xtalk[V_roi]
    #     ax3.plot(pos, data, ''.join(style), label=f'Row:{row} Col:{col}, xtalk={np.nanmax(data): 5.1f}% ')
        
    # ax3.set_xlabel(f'Position (pixel), SN={serial_no}')
    # ax3.set_ylabel( 'Horizontal cross talk (%)')
    # ax3.legend( prop={'family':'monospace', 'size': 8})

    # styles=itertools.product(   ['-', '--', ':'] ,['+', 'o', '*', ',', 'v', 'd', 'x'], 'rgbkycm')

    # pos = np.arange(-V_roi, V_roi+1)
    # ax4=plt.subplot(224)

    # for x, y, xtalk, row, col, style in zip(X, Y, xtalks, Rows, Cols, styles ):
    #     data = xtalk[:,H_roi]
    #     ax4.plot(pos, data, ''.join(style), label=f'Row:{row} Col:{col}, xtalk={np.nanmax(data): 5.1f}% ')

    # ax4.set_xlabel(f'Position (pixel), SN={serial_no}')
    # ax4.set_ylabel( 'Vertical cross talk (%)')
    # ax4.legend( prop={'family':'monospace', 'size': 8}) 
"""

def get_means(image, px, py, Rows, Cols, coverage = 0.25):
    Squares=[]
    
    pitch_y = (py[-1,:].mean()-py[0,:].mean())/Rows
    pitch_x = (px[:,-1].mean()-px[:,0].mean())/Cols
    
    del_x = int(pitch_x*(1-np.sqrt(coverage))/2)
    del_y = int(pitch_y*(1-np.sqrt(coverage))/2)
    
    #print('pitch:', pitch_x, pitch_y, del_x, del_y)

    for i in range(Rows):
        squares=[]
        for j in range(Cols):
            x0, x1 = px[i,j]+del_x, px[i+1,j+1]-del_x
            y0, y1 = py[i,j]+del_y, py[i+1,j+1]-del_y
            
            squares.append( image[ y0:y1,x0:x1].mean())
            
        
        Squares.append(squares)
        
    PD_data=np.array(Squares).ravel()
    
    return PD_data
    
    
def calculate_local(PD_data, px, py, Rows, Cols, checker = True, coverage = 0.25):
    
    pattern = np.indices((Rows,Cols)).sum(axis=0) % 2
    pattern = pattern.astype(bool).ravel()

    sum_squares1 = PD_data[0::2].sum()
    sum_squares2 = PD_data[1::2].sum()

    # +/- checkerboard
    if checker:
        if sum_squares1<sum_squares2:
            pattern = np.logical_not(pattern)
    else:
        pattern[:] = True

    
    adj_squares_list = [ ((i,j+1), (i,j-1),(i-1,j),(i+1,j)) for i in range(Rows) for j in range(Cols)]
    adj_squares = [ list(filter(lambda xy: 0<=xy[0]<Rows and 0<=xy[1]<Cols, pos)) for pos in adj_squares_list] 
    adj_squares = [ np.array(squares).transpose() for squares in adj_squares]

    squares1d =[sq[0]*Cols+sq[1] for sq in adj_squares]




    contrast_array = np.full(Rows*Cols, np.nan)


    for i in range(Rows*Cols):
        if pattern[i]:
            contrast_array[i] =  PD_data[squares1d[i]].mean()/PD_data[i] 
    contrast_array = contrast_array.reshape((Rows, Cols))
    
    return contrast_array


def get_local_from_uniform(image, Rows, Cols, px, py, ax, coverage = 0.25):
    
    PD_data = get_means(image, px, py, Rows, Cols)
    contrast_array = calculate_local(PD_data, px, py, Rows, Cols, False, coverage)
    
    ax1 = ax

    #ax1=fig.add_subplot(211)
    ax1.imshow(image)
    
    pitch_y = (py[-1,:].mean()-py[0,:].mean())/Rows
    pitch_x = (px[:,-1].mean()-px[:,0].mean())/Cols
    
    del_x = int(pitch_x*(1-np.sqrt(coverage))/2)
    del_y = int(pitch_y*(1-np.sqrt(coverage))/2)
    
    for i in range(Rows):
     for j in range(Cols):
         x0, x1 = px[i,j]+del_x, px[i+1,j+1]-del_x
         y0, y1 = py[i,j]+del_y, py[i+1,j+1]-del_y
         
         
         ax1.plot( [px[i,j], px[i+1,j], px[i+1,j+1], px[i,j+1], px[i,j]],
                   [py[i,j], py[i+1,j], py[i+1,j+1], py[i,j+1], py[i,j]],
                   'r-'
                )
         ax1.plot( [x0, x0, x1, x1, x0],[y0,y1, y1, y0, y0],'g-')
    


    contour=ax1.imshow(image, cmap='gray')
    plt.colorbar(contour, ax=ax1)

    
    return contrast_array

def get_local(image, Rows, Cols, px, py, ax, coverage = 0.25):
    PD_data = get_means(image, px, py, Rows, Cols)
    contrast_array = calculate_local(PD_data, px, py, Rows, Cols, coverage)
    
   #fig = plt.figure()

    #ax1=fig.add_subplot(211)
    ax1 = ax
    ax1.imshow(image)
    
    pitch_y = (py[-1,:].mean()-py[0,:].mean())/Rows
    pitch_x = (px[:,-1].mean()-px[:,0].mean())/Cols
    
    del_x = int(pitch_x*(1-np.sqrt(coverage))/2)
    del_y = int(pitch_y*(1-np.sqrt(coverage))/2)
    
    for i in range(Rows):
     for j in range(Cols):
         x0, x1 = px[i,j]+del_x, px[i+1,j+1]-del_x
         y0, y1 = py[i,j]+del_y, py[i+1,j+1]-del_y
         
         
         ax1.plot( [px[i,j], px[i+1,j], px[i+1,j+1], px[i,j+1], px[i,j]],
                   [py[i,j], py[i+1,j], py[i+1,j+1], py[i,j+1], py[i,j]],
                   'r-'
                )
         ax1.plot( [x0, x0, x1, x1, x0],[y0,y1, y1, y0, y0],'g-')
    


    contour=ax1.imshow(image, cmap='gray')
    plt.colorbar(contour, ax=ax1)


    #ax2=fig.add_subplot(212)
    #contour = ax2.imshow(contrast_array)
    #plt.colorbar(contour, ax=ax2)
    
    return contrast_array


    


"""
def find_image_rect():


    im = cv2.imread('test.jpg')
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
"""

import sys
if __name__=='__main__':


    # print(f"Arguments count: {len(sys.argv)}")
    # for i, arg in enumerate(sys.argv):
    #     print(f"Argument {i:>6}: {arg}")

    if len(sys.argv)==4:
        do(sys.argv[1], sys.argv[2], sys.argv[3])

    elif len(sys.argv)==2:
        path = sys.argv[1]
        
        if path[-1] != '/':
            path = path+'/'
            
        
        files = os.listdir(path)
        
        for file in files:
            if file.endswith("pmxm"):
                do(path + file)
        
        print("All files processed!")

    else:
        print('Please input the image file name')
    

