# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 12:24:02 2021

@author: v-olegkozlov

Module to read and decode RIC images
"""

import sqlite3
import numpy as np
import os


class RIC_DB():
    def __init__(self,  database_path):

        connectionObject    = sqlite3.connect(database_path)
        cursorObject        = connectionObject.cursor()
        self.cursorObject = cursorObject
        
    def get_cursor(self):
        return self.cursorObject



    def read_single_channel(self, MeasurementID, Channel = 1):
        """
        Channel should be 0, 1 or 2 (default is 1). 
            Color-coding is tristimulus X(0), Y(1), Z(2)

        Returns
        -------
        image : 2D int32 np.array

        """
        
        cursorObject = self.cursorObject
        
        if Channel not in (0,1,2):
            raise ValueError('Channel should be 0 for X, 1 for Y and 2 for Z')

        query = f"SELECT Width, Height, ImageData FROM ImageData WHERE MeasurementID={MeasurementID} AND DataType={Channel}"

        #query = f"SELECT Width, Height, ImageData FROM ImageData WHERE MeasurementID == {MeasurementID}"#" AND DataType = {Channel}"
                             

        cursorObject.execute(query)
        
        width, height, byte_image = cursorObject.fetchone()
        
        image = np.frombuffer(byte_image, dtype=np.float32).reshape((width, height)).transpose()
        
        return image


    def read_luminance(self, MeasurementID):
    
        # 2nd channel is obsolute luminance
        return self.read_single_channel(MeasurementID, Channel=1)
        

    def get_number_of_measurements(self):
        
        
        self.cursorObject.execute('SELECT Count(*) FROM Measurement')
        length = self.cursorObject.fetchone()[0]
        return length

    def desc_by_index(self, index):
        query = f"""SELECT MeasurementDesc 
                FROM 
                Measurement 
                WHERE 
                MeasurementID = {index}"""
                
        self.cursorObject.execute(query)
        
        try:
            desc = self.cursorObject.fetchone()[0]
        except:
            raise IndexError(f"Index {index} not found in DB")
            
        return desc

        
    def index_by_desc(self, desc):
        query = f"""SELECT MeasurementID 
                FROM 
                Measurement
                WHERE
                MeasurementDesc = '{desc}'
                """
        self.cursorObject.execute(query)
        try:
            index = self.cursorObject.fetchone()[0]
        except:
            raise IndexError(f"{desc} not found in DB")
        return index

    def get_meas_list(self):
        """
        returns {index : desc} dict of all measurements 

        """

        query = f"""SELECT MeasurementID, MeasurementDesc FROM Measurement"""
        
        self.cursorObject.execute(query)
        return self.cursorObject.fetchall()





if __name__ == "__main__":
    
    file = r"E:\OneDrive - Microsoft\RIC\R001_correctCCDorientation.pmxm"
    
    meas = "Ghost_white_50mA"
    
    
    db = RIC_DB(file)
    
    meas_dict = db.get_meas_list()
    print("measurements in the database:\n")
    for key,value in meas_dict:
        print(key, ':', value)
    
    index = db.index_by_desc(meas)
    
    print(f"\nindex for selected measurement is {index}")
    
    image = db.read_luminance(index)
    
    from matplotlib import pyplot as plt
    
    plt.imshow(image)