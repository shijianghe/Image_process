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
from intersect import intersection



def do(database_path, Rows=6, Cols=9):
    try:
        Cols = int(Cols)
        Rows = int(Rows)
    except:
        print('Rows and Cols are numbers')
        raise

    db = RIC_DB(database_path)
    all_measurement_list = db.get_meas_list()

    print('Image database name: ', database_path)
    print(all_measurement_list)

    for MeasurementID, MeasurementDesc in all_measurement_list:
        get_contrast2(db, Rows, Cols, MeasurementID)

    # serial_no = os.path.splitext( os.path.split(database_path)[-1] )[0]



    plt.show()


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


def get_contrast2(db, Rows, Cols, MeasurementID, Bin=8):


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
    
    print(x3,y3,x4,y4)
    
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
    


    fig = plt.figure()

    ax1=fig.add_subplot(211)
    ax1.imshow(image)
    
    px = X.astype(np.int32)
    py = Y.astype(np.int32)
   
    Squares=[]
    
    pitch_y = (Y[-1,:].mean()-Y[0,:].mean())/Rows
    pitch_x = (X[:,-1].mean()-X[:,0].mean())/Cols
    
    coverage = 0.25
    del_x = int(pitch_x*(1-np.sqrt(coverage))/2)
    del_y = int(pitch_y*(1-np.sqrt(coverage))/2)
    
    print('pitch:', pitch_x, pitch_y, del_x, del_y)

    for i in range(Rows):
        squares=[]
        for j in range(Cols):
            x0, x1 = px[i,j]+del_x, px[i+1,j+1]-del_x
            y0, y1 = py[i,j]+del_y, py[i+1,j+1]-del_y
            
            squares.append( image[ y0:y1,x0:x1].mean())
            
            ax1.plot( [X[i,j], X[i+1,j], X[i+1,j+1], X[i,j+1], X[i,j]],
                      [Y[i,j], Y[i+1,j], Y[i+1,j+1], Y[i,j+1], Y[i,j]],
                      'r-'
                   )
            ax1.plot( [x0, x0, x1, x1, x0],[y0,y1, y1, y0, y0],'g-')
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


    contour=ax1.imshow(image, cmap='gray')
    plt.colorbar(contour, ax=ax1)


    ax2=fig.add_subplot(212)
    contour = ax2.imshow(contrast_array)
    plt.colorbar(contour, ax=ax2)



    



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
        do(sys.argv[1], sys.argv[2], sys.argv[3])

    elif len(sys.argv)==2:
        do(sys.argv[1])

    else:
        print('Please input the image file name')
    

