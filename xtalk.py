'''

Author: Shijiang He, shih@microsoft.com
Date:   11/20/2021

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

    # for item in meas_list:
    #     print(item)

    print(f'database name={db_name} threshold={threshold} \n')
    print('serial_no, row, col, cross-talk (pixels)')

    for ID, serial_no in db.get_meas_list():
        # print(f'db={db_name} threshold={threshold} ID={ID}')

        image = db.read_luminance(ID)

        get_xtalk(image, serial_no, threshold)

    plt.show()


def get_xtalk(image, serial_no, threshold):

    threshold_background = 0.10

    H_pix,  V_pix  = 9, 9
    H_roi,  V_roi  = 10, 10
    H_win,  V_win  = H_roi*2+1, V_roi*2+1   # half
    # file_name = r'C:\temp\uLED_Xtalk\cross_talk.png'

    sigma = 20
    img = gaussian_filter(image, sigma)

    H_pix = math.ceil(H_pix)
    V_pix = math.ceil(V_pix)

    peak_ADC=img.max()
    img_bw=img>peak_ADC*threshold_background

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

    
    # print(Rows)
    # print(Cols)
    # print(keys)


    Num_Row = max(Rows)+1
    Num_Col = max(Cols)+1

    xtalk_map_full = np.full((Num_Row*V_win, Num_Col*H_win), np.nan)

    # print(Num_Row, Num_Col, xtalk_map_full.shape)

    # cell = np.full((V_pix+2, H_pix+2), 1.0)

    xtalk_map_list=[]


    fig = plt.figure()

    ax1=fig.add_subplot(221)
    contour = ax1.imshow(img)
    plt.colorbar(contour, ax=ax1)


    H_pos = np.arange(-H_roi, H_roi+1)
    V_pos = np.arange(-V_roi, V_roi+1)

    for x, y, row, col in zip(X, Y, Rows, Cols):

        xx = np.array([int(x+1)-x] + [1]*H_pix + [ x-int(x)] )
        yy = np.array([int(y+1)-y] + [1]*V_pix + [ y-int(y)] )
        cell = np.multiply(xx,yy.reshape((-1,1)))

        
        x0, y0 = int(x), int(y)
        xtalk_map = []
            
        for j in V_pos:
            y_start = y0 + j*V_pix - V_pix//2
            y_end   = y_start + V_pix+2
            
            if y_start<0 or y_end>Ny:
                continue

            for i in H_pos:
                x_start = x0 + i*H_pix - H_pix//2
                x_end   = x_start + H_pix+2
                if x_start<0 or x_end>Nx: 
                    continue
            
                rect = image[y_start: y_end, x_start: x_end]
                xtalk_map.append( (rect*cell).sum() )

                # x1,x2 = x+i*H_pix-H_pix//2, x+i*H_pix+H_pix//2  
                # y1,y2 = y+j*V_pix-V_pix//2, y+j*V_pix+V_pix//2
                # ax1.plot([x1,x1,x2,x2,x1],[y1,y2,y2,y1,y1],'r-', linewidth=1)

        xtalk_map = np.array(xtalk_map).reshape(((V_roi*2+1, H_roi*2+1)))            
        xtalk_map = xtalk_map/xtalk_map.max()*100  # percentage

        # xtalk_map[V_roi, H_roi] = np.nan
        xtalk_map_full[row*V_win:(row+1)*V_win, col*H_win:(col+1)*H_win] = xtalk_map
        xtalk_map_list.append(xtalk_map)





    ax2=fig.add_subplot(222)
    contour = ax2.imshow(xtalk_map_full)
    plt.colorbar(contour, ax=ax2)
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')



    styles=itertools.product(   ['-', '--', ':'] ,['+', 'o', '*', ',', 'v', 'd', 'x'], 'rgbkycm')

    ax3=plt.subplot(223)

    for x, y, xtalk_map, row, col, style in zip(X, Y, xtalk_map_list, Rows, Cols, styles ):
        data = xtalk_map[V_roi]
        
        xtalks_in_pix = np.max(np.abs(zerocross1d(H_pos, data-threshold)))
        ax3.plot(H_pos, data, ''.join(style), label=f'Row:{row} Col:{col}, xtalk={xtalks_in_pix:.1f} pix')
        print(f'{serial_no}, HORI, {row}, {col}, {xtalks_in_pix:.1f}' )
        
    ax3.plot(H_pos, np.full(H_pos.shape, threshold), 'r--')
    ax3.set_xlabel(f'Position (pixel), SN={serial_no}')
    ax3.set_ylabel( 'Horizontal cross talk (%)')
    ax3.legend( prop={'family':'monospace', 'size': 8})

    styles=itertools.product(   ['-', '--', ':'] ,['+', 'o', '*', ',', 'v', 'd', 'x'], 'rgbkycm')

    ax4=plt.subplot(224)

    for x, y, xtalk_map, row, col, style in zip(X, Y, xtalk_map_list, Rows, Cols, styles ):
        data = xtalk_map[:,H_roi]
        xtalks_in_pix = np.max(np.abs(zerocross1d(V_pos, data-threshold)))

        ax4.plot(V_pos, data, ''.join(style), label=f'Row:{row} Col:{col}, xtalk={xtalks_in_pix:.1f} pix')
        print(f'{serial_no}, VERT, {row}, {col}, {xtalks_in_pix:.1f}' )

    ax4.plot(V_pos, np.full(V_pos.shape, threshold), 'r--')
    ax4.set_xlabel(f'Position (pixel), SN={serial_no}')
    ax4.set_ylabel( 'Vertical cross talk (%)')
    ax4.legend( prop={'family':'monospace', 'size': 8}) 






def zerocross1d(x, y, getIndices=False):
  """
    https://github.com/sczesla/PyAstronomy/blob/master/src/pyaC/mtools/zerocross.py

    Find the zero crossing points in 1d data.
    
    Find the zero crossing events in a discrete data set.
    Linear interpolation is used to determine the actual
    locations of the zero crossing between two data points
    showing a change in sign. Data point which are zero
    are counted in as zero crossings if a sign change occurs
    across them. Note that the first and last data point will
    not be considered whether or not they are zero. 
    
    Parameters
    ----------
    x, y : arrays
        Ordinate and abscissa data values.
    getIndices : boolean, optional
        If True, also the indicies of the points preceding
        the zero crossing event will be returned. Defeualt is
        False.
    
    Returns
    -------
    xvals : array
        The locations of the zero crossing events determined
        by linear interpolation on the data.
    indices : array, optional
        The indices of the points preceding the zero crossing
        events. Only returned if `getIndices` is set True.
  """
  
  # Check sorting of x-values
  if np.any((x[1:] - x[0:-1]) <= 0.0):
    raise(PE.PyAValError("The x-values must be sorted in ascending order!", \
                         where="zerocross1d", \
                         solution="Sort the data prior to calling zerocross1d."))
  
  # Indices of points *before* zero-crossing
  indi = np.where(y[1:]*y[0:-1] < 0.0)[0]
  
  # Find the zero crossing by linear interpolation
  dx = x[indi+1] - x[indi]
  dy = y[indi+1] - y[indi]
  zc = -y[indi] * (dx/dy) + x[indi]
  
  # What about the points, which are actually zero
  zi = np.where(y == 0.0)[0]
  # Do nothing about the first and last point should they
  # be zero
  zi = zi[np.where((zi > 0) & (zi < x.size-1))]
  # Select those point, where zero is crossed (sign change
  # across the point)
  zi = zi[np.where(y[zi-1]*y[zi+1] < 0.0)]
  
  # Concatenate indices
  zzindi = np.concatenate((indi, zi)) 
  # Concatenate zc and locations corresponding to zi
  zz = np.concatenate((zc, x[zi]))
  
  # Sort by x-value
  sind = np.argsort(zz)
  zz, zzindi = zz[sind], zzindi[sind]
  
  if not getIndices:
    return zz
  else:
    return zz, zzindi





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
    

