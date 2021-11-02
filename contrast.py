'''

Author: Shijiang He, shih@microsoft.com


'''


import cv2
from PIL import Image 

import numpy as np

import itertools, math, os
import matplotlib.pyplot as plt

from checkerboard import detect_checkerboard 
from RIC import RIC_DB

def contrast(file_name, Cols, Rows, binning=2):

    coverage = 0.5

    db = RIC_DB(file_name)
    image = db.read_luminance(1)
    image = image.reshape(3264//8, 8, 4896//8, 8).mean(-1).mean(1)    

    try:
        Cols = int(Cols)
        Rows = int(Rows)
    except:
        print('Rows and Cols are numbers')
        raise

    print('image file name: ', file_name)


    rows = Rows-1; ROWS=Rows+1
    cols = Cols-1; COLS=Cols+1

    # image = cv2.imread(file_name,cv2.IMREAD_GRAYSCALE)
    serial_no = os.path.splitext( os.path.split(file_name)[-1] )[0]
    # image = np.asarray(image.convert('L')).astype(np.uint16)


    corners, score = detect_checkerboard(image, (cols,rows))

    if score >0.9:
        print( f'Checkerboard is not found in {serial_no}')
        return False 


    x=corners[:,0,1].reshape((cols,rows)).transpose()
    y=corners[:,0,0].reshape((cols,rows)).transpose()

    X = np.full((ROWS, COLS), np.nan)
    Y = np.full((ROWS, COLS), np.nan)
    X[1:-1,1:-1] = x
    Y[1:-1,1:-1] = y

    Y[ 0,1:-1]=y[ 0,:]; X[ 0,1:-1]= 2*x[ 0,:]-x[ 1,:]
    Y[-1,1:-1]=y[-1,:]; X[-1,1:-1]= 2*x[-1,:]-x[-2,:]
    X[1:-1, 0]=x[:, 0]; Y[1:-1, 0]= 2*y[ :,0]-y[ :,1]
    X[1:-1,-1]=x[:,-1]; Y[1:-1,-1]= 2*y[:,-1]-y[:,-2]

    Y[ 0, 0] = Y[ 1, 0];  X[ 0, 0]=X[ 0, 1]
    Y[-1, 0] = Y[-2, 0];  X[-1, 0]=X[-1, 1]
    Y[-1,-1] = Y[-2,-1];  X[-1,-1]=X[-1,-2]
    Y[ 0,-1] = Y[ 1,-1];  X[ 0,-1]=X[ 0,-2]


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

    plt.show()




def find_image_rect():


    im = cv2.imread('test.jpg')
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

import sys
if __name__=='__main__':


    # print(f"Arguments count: {len(sys.argv)}")
    # for i, arg in enumerate(sys.argv):
    #     print(f"Argument {i:>6}: {arg}")

    if len(sys.argv)==4:
        contrast(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print('Please input the image file name')
    

