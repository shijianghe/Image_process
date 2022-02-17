# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 16:47:07 2021

@author: v-olegkozlov

Run via command line python contrast_v2.py [path] [args]

e.g. python contrast_v2.py L002.pmxm C:/RIC L001.pmxm -R001.pmxm bin=16 check_roi = 0.3 save_all plot
(replace double slash with a single slash in the file path)

path (optional): path to folder with data, if not present default path is used

args (any order):
    
    DB.ttxm : DB to process (any number of files, if absent all files in 
                               the folder will be processed)
    
    -DB.pmxm : DB to exclude (any number of files, if absent all files in 
                               the folder will be processed)
    
    bin=X : (e.g. bin=16) defines bin size. Default is 8. Bin size of less than 
                                    4 is not recommended (too heavy to process)
                                
                                            
    save_all or s: if present, DOES NOT save extended data (full image matrices). 
                            Saving extended data may ignificantly increase
                             file size and processing time if binning is <4
"""


from RIC import RIC_DB
import image_processing
import matplotlib.pyplot as plt
import os

import math

import sys

import pandas as pd


default_path = r"E:/OneDrive - Microsoft/uLED"


RIC_extensiton = ".ttxm"

save_full = True
bin_size = 8



stats_from_full_image = False


dpi = 300


channels = ['R', 'G', 'B', 'W']
modes = ['+chkbrd', '-chkbrd', 'white', 'black']
suffixex = ["50mA"]

fig_height = 10

fig_cols = ['+Check Raw', '-Check Raw', 'Check Local', 'Check Seq', 
            'White Uniformity', 'Black Uniformity', 'W/B Seq', 'ANSI Raw', 'ANSI uniformity']


def process_data(database_path):
    db = RIC_DB(database_path)
    measurements_in_db = db.get_meas_list()
    
    measurements_to_process = []
    
    for measurement in measurements_in_db:
        if "frame" not in measurement[1].lower():
            measurements_to_process.append(measurement[1])

    
    save_dir = database_path.split('.')[0] + "/"
    
    DMA_name = save_dir.split('/')[-2]
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        

    print('Image database name: ', database_path)
    print('Measurements in this DB:')
    print(measurements_in_db)
 
    print('Measurements to be processed DB:')
    print(measurements_to_process)   
 
    data = process_all_measurements(db, measurements_to_process)
    save_data(data, save_dir, DMA_name)
    plot_data(data, save_dir,  DMA_name)
    print ("All measurements are processed!")
    

def save_data(data, save_dir, DMA_name):
    print("Saving data")
    file = save_dir + f"results_{DMA_name}.xlsx"
    writer = pd.ExcelWriter(file,  engine='xlsxwriter')
    
    write_stats_to_excel(writer, data)
    
    if save_full:
        for measurement in data.keys():
            write_data_to_excel(writer, data, measurement, "", "luminance_uniformity","")
         
    writer.close()
    print("Data saved!")

def plot_data(data,save_dir, DMA_name):
    print("Plotting data")
    fig, axs, nrow, ncol  = prepare_figure(data, DMA_name)
    
    for i, key in enumerate(data.keys()):
        
        current_row = i // ncol
        current_col = i % ncol
        
        if nrow > 1 and ncol > 1:
            ax = axs[current_row,current_col]
        elif ncol > 1:
            ax = axs[current_col]
        else:
            ax = axs
            
            
        image = data[key]['luminance_uniformity']
        stats = data[key]['luminance_uniformity_stats']
        
        contour=ax.imshow(image, cmap='gray')

        contour=ax.imshow(image)
        plt.colorbar(contour, ax=ax)
        ax.set_xlabel(f"""min: {stats[0]:.1f} max: {stats[1]:.1f} avg: {stats[2]:.1f} SD {stats[3]:.1f}""")
        ax.title.set_text(key)            
     
    plt.tight_layout()
    plt.savefig(save_dir+f'plot_{DMA_name}.png', dpi=dpi, bbox_inches='tight')
    
    plt.show()
        

                   


        

def prepare_figure(data, DMA_name):
    total_figs = len(data.keys())
    
    fig_cols = math.ceil(math.sqrt(total_figs))
    fig_rows = math.ceil(total_figs/fig_cols)
    
    
    fig, axs = plt.subplots(fig_rows, fig_cols)
    
    fig.set_figheight(fig_height)
    fig.set_figwidth(fig_height * fig_cols/fig_rows)
        
    fig.suptitle(DMA_name, size=16)
    return fig,axs, fig_rows, fig_cols

    
def write_stats_to_excel(writer,data):

    measurements = list(data.keys())
    
    df = pd.DataFrame()
    
    for measurement in measurements:
        df[measurement] = data[measurement]['luminance_uniformity_stats']
    
    df = df.set_axis(['min', 'max', 'avg', 'SD'], axis = 0)
    
    
    df.to_excel(writer, sheet_name = 'stats', na_rep = 'NaN')


def write_data_to_excel(writer, data, base, data_type, data_ID,sheet_suffix):
    df = pd.DataFrame(data[base+data_type][data_ID])
    df.to_excel(writer, sheet_name = base+sheet_suffix,
                na_rep = 'NaN', index = False, header=False)
    worksheet = writer.sheets[base+sheet_suffix]
    worksheet.conditional_format(0, 0, 2500, 2500,
                             {'type': '3_color_scale'
                             })
    
def process_all_measurements(db, measurements_to_process):  
    data = {}
    
    for i, measurement in enumerate(measurements_to_process):
        data[measurement] = process_measurement(db, measurement)
                  
    return data

def process_measurement(db, measurement):
    try:
        print(f"Processing {measurement}")
        measurement_ID = db.index_by_desc(measurement)
    except:
        print(f"Measurement {measurement} cannot be processed")
        return 
    
    luminance = db.read_luminance(measurement_ID)
    luminance_binned = image_processing.pre_process_uLED_image(luminance, bin_size)
    luminance_uiformity_binned = image_processing.calculate_uniformity(luminance_binned)
    
    if stats_from_full_image:
        luminance_for_stat = image_processing.pre_process_uLED_image(luminance, 1)
        luminance_uniformity = image_processing.calculate_uniformity(luminance_for_stat)
        stats = image_processing.calculate_statistics(luminance_uniformity)
        
    else:
        stats = image_processing.calculate_statistics(luminance_uiformity_binned)
    
    
    
    

    
    
    result = {"luminance_uniformity" : luminance_uiformity_binned,
              "luminance_uniformity_stats" : stats
              }
    
    return result
        


    
if __name__ == "__main__":

   
    files_to_process = []
    files_to_exclude = []
    
    args = sys.argv
    if "save_all" in args:
        save_full = False
        args.remove("save_all")

    if "s" in args:
        save_full = False
        args.remove("s")
    

        
    bin_size_string = next((s for s in args if "bin=" in s), None)
    
    if bin_size_string:
        try:
            bin_size = int(bin_size_string[4:])
        except:
            print("Cannot parse bin size")
        args.remove(bin_size_string)
    

    
    if len(args) > 1:
        path = args[1]
        if os.path.exists(path):
            args.remove(path)
        else:
            path = default_path
    else:
        path = default_path
        
    if not os.path.exists(path):
        raise FileNotFoundError('Please input valid file name or path or check default path')
    
    
    if len(args) > 1:
        remaining_args = args[1:].copy()
        for arg in remaining_args:
            if arg.endswith(RIC_extensiton):
                if arg[0] == '-':
                    files_to_exclude.append(arg[1:])
                else:
                    files_to_process.append(arg)
            else:
                print(f"argument {arg} is not recognized!")
            args.remove(arg)
                
    

        
    if path.endswith(RIC_extensiton):
        path, filename = os.path.split(path)
        files_to_process.append(filename)
                
    if path[-1] != '/':
        path = path+'/'
        
    if not files_to_process:
        files_to_process = os.listdir(path)
    
    if files_to_exclude:
        for file in files_to_exclude:
            try:
                files_to_process.remove(file)
            except:
                pass
    
    for file in files_to_process:
        if file.endswith(RIC_extensiton):
            print(f"Processing {file}")
            try:
                process_data(path + file)
            except:
                print(f"{file} is not processed!")
        
    print("All files processed!")        

