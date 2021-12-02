# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 16:47:07 2021

@author: v-olegkozlov
"""

from RIC import RIC_DB
import image_tools
import image_processing
import matplotlib.pyplot as plt
import os

import numpy as np

import pandas as pd

import seaborn as sns

import sys


save_full_wb = False

k_x = 0.03
k_y = 0.06

bin_size = 8

dpi = 300

area = 0.6

center_FOV = (10,10)

cols = 9
rows = 6

padding_wb = 5 #in percent

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

fig_height = 10

fig_cols = ['+Check Raw', '-Check Raw', 'Check Local', 'Check Seq', 
            'White Uniformity', 'Black Uniformity', 'W/B Seq', 'ANSI Raw', 'ANSI uniformity']

def process_data(database_path):
    db = RIC_DB(database_path)
    measurements_in_db = db.get_meas_list()
    
    save_dir = database_path.split('.')[0] + "/"
    
    DMA_name = save_dir.split('/')[-2]
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        

    print('Image database name: ', database_path)
    print('Measurements in this DB:')
    print(measurements_in_db)
    
    data = process_all_measurements(db)
    save_data(data, save_dir, DMA_name)
    plot_data(data, save_dir,  DMA_name)
    print ("All measurements are processed!")
    

def save_data(data, save_dir, DMA_name):
    print("Saving data")
    file = save_dir + f"results_{DMA_name}.xlsx"
    writer = pd.ExcelWriter(file,  engine='xlsxwriter')
    for suffix in suffixex:
        for channel in channels:
            base = channel + '_' + suffix + '_'
            write_data_to_excel(writer, data, base, 'check', 'local_contrast', 'check_local')
            write_data_to_excel(writer, data, base, 'check', 'seq_contrast', 'check_sequential')
            write_data_to_excel(writer, data, base, 'wb', 'ansi_uniformity', 'ansi_uniformity') 
            if save_full_wb:
                write_data_to_excel(writer, data, base, 'wb', 'seq_contrast', 'wb_sequential')
                write_data_to_excel(writer, data, base, 'wb', 'white_uniformity', 'white_uniformity')
                write_data_to_excel(writer, data, base, 'wb', 'black_uniformity', 'black_uniformity')
                
            write_stats_to_excel(writer, data, base)

    
    writer.close()
    print("Data saved!")

def plot_data(data,save_dir, DMA_name):
    print("Plotting data")
    fig, axs = prepare_figure(DMA_name)
    for suffix in suffixex:
        for row, channel in enumerate(channels):
            base = channel + '_' + suffix + '_'
            

            ax = axs[row,0]
            image = data[base+'check']['+Check Raw']
            contour=ax.imshow(image, cmap='gray')
            plt.colorbar(contour, ax=ax)
            draw_rois(ax, data[base+'check']['rois'])
            
            ax = axs[row,1]
            image = data[base+'check']['-Check Raw']
            contour=ax.imshow(image, cmap='gray')
            plt.colorbar(contour, ax=ax)
            draw_rois(ax, data[base+'check']['rois'])
            
            ax = axs[row,2]
            image = data[base+'check']['local_contrast']
            stats = data[base+'check']['local_contrast_stats']
            contour=ax.imshow(image)
            plt.colorbar(contour, ax=ax)
            ax.set_xlabel(f"""min: {stats[0]:.1f} max: {stats[1]:.1f} avg: {stats[2]:.1f} SD {stats[3]:.1f}""")
                    
        
            ax = axs[row,3]
            image = data[base+'check']['seq_contrast']
            stats = data[base+'check']['seq_contrast_stats']
            avg2 = data[base+'check']['avg2']
            contour=ax.imshow(image)
            plt.colorbar(contour, ax=ax)
            ax.set_xlabel(f"""min: {stats[0]:.1f} max: {stats[1]:.1f} avg: {stats[2]:.1f} SD {stats[3]:.1f} avg2 {avg2:.1f}""")

            ax = axs[row,4]
            image = data[base+'wb']['white_uniformity']
            stats = data[base+'wb']['white_uniformity_stats']
            contour=ax.imshow(image)
            plt.colorbar(contour, ax=ax)
            ax.set_xlabel(f"""min: {stats[0]:.1f} max: {stats[1]:.1f} avg: {stats[2]:.1f} SD {stats[3]:.1f}""")

            ax = axs[row,5]
            image = data[base+'wb']['black_uniformity']
            stats = data[base+'wb']['black_uniformity_stats']
            contour=ax.imshow(image)
            plt.colorbar(contour, ax=ax)
            ax.set_xlabel(f"""min: {stats[0]:.1f} max: {stats[1]:.1f} avg: {stats[2]:.1f} SD {stats[3]:.1f}""")

            ax = axs[row,6]
            image = data[base+'wb']['seq_contrast']
            stats = data[base+'wb']['seq_contrast_stats']
            contour=ax.imshow(image)
            plt.colorbar(contour, ax=ax)
            ax.set_xlabel(f"""min: {stats[0]:.1f} max: {stats[1]:.1f} avg: {stats[2]:.1f} SD {stats[3]:.1f}""")

            ax = axs[row,7]
            image = data[base+'wb']['white_raw']
            contour=ax.imshow(image)
            plt.colorbar(contour, ax=ax)
            draw_rois(ax, data[base+'wb']['edge_rois'], col = 'b')
            draw_rois(ax, data[base+'wb']['ANSI_rois'])
            
            ax = axs[row,8]
            image = data[base+'wb']['ansi_uniformity']
            stats = data[base+'wb']['ansi_uniformity_stats']
            sns.heatmap(image, annot = True, fmt = ".2f", 
                        xticklabels=False, yticklabels=False, cmap='viridis', ax = ax)

            ax.set_xlabel(f"""min: {stats[0]:.1f} max: {stats[1]:.1f} avg: {stats[2]:.1f} SD {stats[3]:.1f}""")
    
    plt.tight_layout()
    plt.savefig(save_dir+f'plot_{DMA_name}.png', dpi=dpi, bbox_inches='tight')
    plt.show()
        

                   
            
def draw_rois(ax,rois, col = 'r'):
    rois = np.array(rois)
    rois =rois.reshape(-1, rois.shape[-1])
    for i in range(rois.shape[0]):
        coord = rois[i]
        ax.plot( [coord[0], coord[0], coord[2], coord[2], coord[0]],[coord[1],coord[3], coord[3], coord[1], coord[1]],color = col)
    
            

        

def prepare_figure(DMA_name):
    fig_rows = channels
    fig, axs = plt.subplots(len(fig_rows), len(fig_cols))
    
    fig.set_figheight(fig_height)
    fig.set_figwidth(fig_height * len(fig_cols)/len(fig_rows) * 1.5)
    
    for i, row in enumerate(fig_rows):
        axs[i,0].set_ylabel(row)
    
    for i, title in enumerate(fig_cols):
        axs[0,i].set_title(title)
        
    fig.suptitle(DMA_name, size=16)
    return fig,axs

    
def write_stats_to_excel(writer,data, base):
    df_stats = pd.DataFrame(data[base+'check']['local_contrast_stats'],columns = ['check_local']).T
    df_stats = df_stats.append(
        pd.DataFrame(data[base+'check']['seq_contrast_stats'],columns = ['check_seq']).T)
    df_stats = df_stats.append(
        pd.DataFrame(data[base+'wb']['white_uniformity_stats'],columns = ['white_uniformity']).T)
    df_stats = df_stats.append(
        pd.DataFrame(data[base+'wb']['white_uniformity_luminance_stats'],columns = ['white_uniformity_luminance']).T)
    df_stats = df_stats.append(
        pd.DataFrame(data[base+'wb']['black_uniformity_stats'],columns = ['black_uniformity']).T)
    df_stats = df_stats.append(
        pd.DataFrame(data[base+'wb']['black_uniformity_luminance_stats'],columns = ['black_uniformity_luminance']).T)
    df_stats = df_stats.append(
        pd.DataFrame(data[base+'wb']['seq_contrast_stats'],columns = ['seq_contrast']).T)
    df_stats = df_stats.append(
        pd.DataFrame(data[base+'wb']['ansi_uniformity_stats'],columns = ['ansi_uniformity']).T)
    df_stats = df_stats.append(
        pd.DataFrame(data[base+'wb']['white_center_luminance_stats'],columns = ['white_lumimance_center']).T)
    
    df_stats.columns = ['min','max','avg','SD']
    
    df_stats['avg2'] = ""
    
    df_stats.at['check_seq', 'avg2'] = data[base+'check']['avg2']
    
    df_stats.to_excel(writer, sheet_name = base+"stats",
                na_rep = 'NaN')

def write_data_to_excel(writer, data, base, data_type, data_ID,sheet_suffix):
    df = pd.DataFrame(data[base+data_type][data_ID])
    df.to_excel(writer, sheet_name = base+sheet_suffix,
                na_rep = 'NaN', index = False, header=False)
    worksheet = writer.sheets[base+sheet_suffix]
    worksheet.conditional_format(0, 0, 100, 100,
                             {'type': '3_color_scale'
                             })
    
def process_all_measurements(db):  
    data = {}
    baseline_image_ID = db.index_by_desc(f"W_white_{suffixex[0]}")
    baseline_image = db.read_luminance(baseline_image_ID)
    angle, corners, corners_2 = image_processing.get_parameters_from_white(baseline_image, bin_size, k_x, k_y)
    
    for k, suffix in enumerate(suffixex):
        for i, channel in enumerate(channels):
                current_index = channel + "_" + suffix               
                check_results =  process_checkers(db, suffix, channel, angle, corners, corners_2)  
                data[current_index+"_check"] = check_results
                
                wb_results = process_white_black(db, suffix, channel, angle, corners, corners_2)
                data[current_index+"_wb"] = wb_results
                
    
    return data


def process_white_black(db, suffix, channel, angle, corners, corners_2):
    measurement_white = f"{channel}_white_{suffix}"
    measurement_black = f"{channel}_black_{suffix}"
    
    try:
        print(f"Processing {measurement_white}")
        measurement_white_ID = db.index_by_desc(measurement_white)
    except:
        print(f"Measurement {measurement_white} not in DB")
        return 
    
    try:
        print(f"Processing {measurement_black}")
        measurement_black_ID = db.index_by_desc(measurement_black)
    except:
        print(f"Measurement {measurement_black} not in DB")
        return   
    
    white = db.read_luminance(measurement_white_ID)
    black = db.read_luminance(measurement_black_ID)
    
    white = image_processing.pre_process_image(white, angle, corners, corners_2, k_x, k_y, bin_size)
    black = image_processing.pre_process_image(black, angle, corners, corners_2, k_x, k_y, bin_size)
    
    white -= np.min(white)
    black -= np.min(black)
    
    white[white == 0] = np.min(white[white!=0])
    black[black == 0] = np.min(black[black!=0])
    
    padding_V = int(white.shape[0]*padding_wb/200)
    padding_H = int(white.shape[1]*padding_wb/200)
    
    white = white[padding_V:white.shape[0] - padding_V, padding_H:white.shape[1] - padding_H]
    black = black[padding_V:black.shape[0] - padding_V, padding_H:black.shape[1] - padding_H]
    
    white_center_roi = image_tools.get_roi_from_angles(white, (0,0), angle_roi=center_FOV)
    white_center = image_tools.crop_image(white, white_center_roi)
    
    white_uniformity = image_processing.calculate_uniformity(white)
    black_uniformity = image_processing.calculate_uniformity(black)
    
    wb_seq = image_processing.calculate_sequential_WB_contrast(white, black)
    
    ansi_uniformity, ansi_roi_array, edge_roi_array = image_processing.calculate_ansi(white, ANSI, edges)
    
    result = {"white_uniformity" : white_uniformity,
              "black_uniformity" : black_uniformity,
              "seq_contrast": wb_seq,
              "ansi_uniformity":ansi_uniformity,
              "white_uniformity_stats" : image_processing.calculate_statistics(white_uniformity),
              "black_uniformity_stats" : image_processing.calculate_statistics(black_uniformity),
              "white_uniformity_luminance_stats" : image_processing.calculate_statistics(white),
              "black_uniformity_luminance_stats" : image_processing.calculate_statistics(black),
              "white_center_luminance_stats" : image_processing.calculate_statistics(white_center),
              "seq_contrast_stats": image_processing.calculate_statistics(wb_seq),
              "ansi_uniformity_stats":image_processing.calculate_statistics(ansi_uniformity),
              "ANSI_rois":ansi_roi_array,
              "edge_rois":edge_roi_array,
              "white_raw":white}

    
    return result
    
    
def process_checkers(db, suffix, channel, angle, corners, corners_2):
    measurement_plus = f"{channel}_+chkbrd_{suffix}"
    measurement_minus = f"{channel}_-chkbrd_{suffix}"
    
    try:
        print(f"Processing {measurement_plus}")
        measurement_plus_ID = db.index_by_desc(measurement_plus)
    except:
        print(f"Measurement {measurement_plus} not in DB")
        return 
    
    try:
        print(f"Processing {measurement_minus}")
        measurement_minus_ID = db.index_by_desc(measurement_minus)
    except:
        print(f"Measurement {measurement_minus} not in DB")
        return   
    
    check_plus = db.read_luminance(measurement_plus_ID)
    check_minus = db.read_luminance(measurement_minus_ID)
    
    check_plus = image_processing.pre_process_image(check_plus, angle, corners, corners_2, k_x, k_y, bin_size)
    check_minus = image_processing.pre_process_image(check_minus, angle, corners, corners_2, k_x, k_y, bin_size)
    
    squares = image_tools.get_check_rois(check_plus, area = area)
    
    intensities = image_tools.get_intensities(check_plus, squares)
    intensities2 = image_tools.get_intensities(check_minus, squares)

    contrast_combined = image_processing.calculate_local_checkerboard_contrast(intensities,intensities2)
    
    contrast_seq = image_processing.calculate_sequential_checkerboard_contrast(intensities,intensities2)
    
    pattern_plus = image_tools.generate_pattern(rows, cols, "+")
    pattern_minus = image_tools.generate_pattern(rows, cols, '-')
    
    whites_mean = np.hstack((intensities[pattern_minus], 
                             intensities2[pattern_plus])).mean()
    
    blacks_mean = np.hstack((intensities[pattern_plus], 
                             intensities2[pattern_minus])).mean()  
    
    avg2 = whites_mean/blacks_mean
    
    result = {"+Check Raw" : check_plus,
              "-Check Raw" : check_minus,
              "local_contrast": contrast_combined,
              "local_contrast_stats": image_processing.calculate_statistics(contrast_combined),
              "seq_contrast": contrast_seq,
              "seq_contrast_stats": image_processing.calculate_statistics(contrast_seq),
              "rois": squares,
              "avg2":avg2
        }
    
    return result

    
if __name__ == "__main__":

    if len(sys.argv)==2:
        path = sys.argv[1]
        
        if os.path.isfile(path):
            process_data(path)
        else:
        
            if path[-1] != '/':
                path = path+'/'
                            
            files = os.listdir(path)
            
            for file in files:
                if file.endswith("pmxm"):
                    print(f"Processing {file}")
                    try:
                        process_data(path + file)
                    except:
                        print(f"{file} is not processed!")
            
            print("All files processed!")        

    else:
        print('Please input the image file name or path')