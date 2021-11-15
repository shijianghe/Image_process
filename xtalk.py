'''

Author: Shijiang He, shih@microsoft.com


'''


from PIL import Image 
import numpy as np
from scipy import ndimage
import itertools, math, os
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

from RIC import RIC_DB


def do(db_name, threshold):

    threshold = float(threshold)

    db = RIC_DB(db_name)
    meas_list = db.get_meas_list()

    for item in meas_list:
        print(item)

    for ID, serial_no in db.get_meas_list():
        print(f'db={db_name} threshold={threshold} ID={ID}')

        image = db.read_luminance(ID)

        xtalk(image, serial_no, threshold)

    plt.show()


def xtalk(image, serial_no, threshold):


    H_pix,  V_pix  = 9, 9
    H_roi,  V_roi  = 10, 10
    H_win,  V_win  = H_roi*2+1, V_roi*2+1   # half
    # file_name = r'C:\temp\uLED_Xtalk\cross_talk.png'

    sigma = 20
    img = gaussian_filter(image, sigma)

    H_pix = math.ceil(H_pix)
    V_pix = math.ceil(V_pix)

    peak_ADC=img.max()
    img_bw=img>peak_ADC*threshold

    labeled_array, num_features = ndimage.label(img_bw)
    Ny, Nx=img.shape
    X_grid, Y_grid=np.meshgrid(np.arange(Nx), np.arange(Ny))

    X, Y = [], []
    for i in range(1, num_features+1):
        idx = (labeled_array==i)
        xi = (X_grid[idx]*img[idx]).sum()/img[idx].sum()
        yi = (Y_grid[idx]*img[idx]).sum()/img[idx].sum()
        X.append(xi)
        Y.append(yi)

    X=np.array(X)
    Y=np.array(Y)
    X0 = X.min()
    Y0 = Y.min()

    distances = [ np.sqrt((x1-x2)**2+(y1-y2)**2) for (x1,y1),(x2,y2) in itertools.combinations(zip(X,Y),2)]
    min_dist = min(distances)


    Rows = [round((y-Y0)/min_dist) for y in Y]
    Cols = [round((x-X0)/min_dist) for x in X]

    keys = [f'{int(row):4d}-{int(col):4d}' for row, col in zip(Rows, Cols)]
    info = [[ x, y, row, col] for _, x,y, row, col in sorted(zip(keys, X, Y, Rows, Cols))]
    X,Y,Rows, Cols = list(zip(*info))
    # print(keys,X,Y)

    
    print(Rows)
    print(Cols)
    print(keys)

    

    Num_Row = max(Rows)+1
    Num_Col = max(Cols)+1

    xtalk_map = np.full((Num_Row*V_win, Num_Col*H_win), np.nan)

    print(Num_Row, Num_Col, xtalk_map.shape)

    cell = np.full((V_pix+2, H_pix+2), 1.0)

    xtalks=[]


    fig = plt.figure()

    ax1=fig.add_subplot(221)
    contour = ax1.imshow(img)
    plt.colorbar(contour, ax=ax1)


    for x, y, row, col in zip(X, Y, Rows, Cols):
        Res_x, Res_y = x-int(x), y-int(y)
        res_x, res_y = 1-Res_x, 1-Res_y
        cell[-1, :] = Res_y
        cell[ 0, :] = res_y
        cell[ :,-1] = Res_x
        cell[ :, 0] = res_x
        cell[ 0, 0] = res_y*res_x
        cell[ 0,-1] = res_y*Res_x
        cell[-1, 0] = Res_y*res_x
        cell[-1,-1] = Res_y*Res_x

        
        x0, y0 = int(x), int(y)
        xtalk = np.zeros((V_roi*2+1, H_roi*2+1))
        
        for i in range(-H_roi, H_roi):
            x_start = x0 + i*H_pix - H_pix//2
            x_end   = x_start + H_pix+2
            if x_start<0 or x_end>Nx: 
                continue
            
            for j in range(-V_roi, V_roi):
                y_start = y0 + j*V_pix - V_pix//2
                y_end   = y_start + V_pix+2
                
                if y_start<0 or y_end>Ny:
                    continue
            
                rect = image[y_start: y_start+V_pix+2, x_start: x_start+H_pix+2]
                val = (rect*cell).sum()
                xtalk[j+V_roi, i+H_roi]=(rect*cell).sum()

                # x1,x2 = x+i*H_pix-H_pix//2, x+i*H_pix+H_pix//2  
                # y1,y2 = y+j*V_pix-V_pix//2, y+j*V_pix+V_pix//2
                # ax1.plot([x1,x1,x2,x2,x1],[y1,y2,y2,y1,y1],'r-', linewidth=1)

            
        xtalk = xtalk/xtalk.max()*100  # percentage

        # row = round(y/min_dist)
        # col = round(x/min_dist)

        xtalk[V_roi, H_roi] = np.nan
        xtalk_map[row*V_win:(row+1)*V_win, col*H_win:(col+1)*H_win] = xtalk
        xtalks.append(xtalk)





    ax2=fig.add_subplot(222)
    contour = ax2.imshow(xtalk_map)
    plt.colorbar(contour, ax=ax2)
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')



    styles=itertools.product(   ['-', '--', ':'] ,['+', 'o', '*', ',', 'v', 'd', 'x'], 'rgbkycm')

    pos = np.arange(-H_roi, H_roi+1)
    ax3=plt.subplot(223)

    for x, y, xtalk, row, col, style in zip(X, Y, xtalks, Rows, Cols, styles ):
        data = xtalk[V_roi]
        ax3.plot(pos, data, ''.join(style), label=f'Row:{row} Col:{col}, xtalk={np.nanmax(data): 5.1f}% ')
        
    ax3.set_xlabel(f'Position (pixel), SN={serial_no}')
    ax3.set_ylabel( 'Horizontal cross talk (%)')
    ax3.legend( prop={'family':'monospace', 'size': 8})

    styles=itertools.product(   ['-', '--', ':'] ,['+', 'o', '*', ',', 'v', 'd', 'x'], 'rgbkycm')

    pos = np.arange(-V_roi, V_roi+1)
    ax4=plt.subplot(224)

    for x, y, xtalk, row, col, style in zip(X, Y, xtalks, Rows, Cols, styles ):
        data = xtalk[:,H_roi]
        ax4.plot(pos, data, ''.join(style), label=f'Row:{row} Col:{col}, xtalk={np.nanmax(data): 5.1f}% ')

    ax4.set_xlabel(f'Position (pixel), SN={serial_no}')
    ax4.set_ylabel( 'Vertical cross talk (%)')
    ax4.legend( prop={'family':'monospace', 'size': 8}) 



import sys
if __name__=='__main__':


    # print(f"Arguments count: {len(sys.argv)}")
    # for i, arg in enumerate(sys.argv):
    #     print(f"Argument {i:>6}: {arg}")

    argv = sys.argv
    if len(argv)==3:
        do(argv[1], argv[2])
    else:
        print('Please input the image file name')
    

