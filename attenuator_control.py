# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 09:37:10 2021

@author: v-olegkozlov
"""

import serial
import time
import math
import re
import random

class attenuator_control():
    """
    Drives attenuator with Arduino board
    Desined to be used with step_motor_manual_test sketch
    
    Units are 1/2 steps 
    
    Total range is ~1200, ND out is 0, ND in is 1200, home is ~850
    """
    
    def __init__(self, COM, baudrate = 9600, step_size = 100, travel_range = 1200, home_on_startup = True):
        if not isinstance(COM, int):
            raise TypeError("COM should be integer!")
            


        self.port = f"COM{COM}"
        self.baudrate = int(baudrate)
        self.step_size = int(step_size)
        self.travel_range = int(travel_range)
        self.current_position = None
        self.home_position = None
        
        self.is_open = False
        self.is_homed = False
        
        if home_on_startup:
            self.home()
        
        print('Init done')
        
        
        
    def open_serial(self):
        print(f'Opening {self.port}')
        
        self.ser = serial.Serial(self.port, self.baudrate, timeout = 1)
        
        time.sleep(3)
        output = self.read_serial()
        
        if not "setup for DMA Module" in output:
            raise Exception("Serial is not open!")
        else:
            print("Serial open")
            self.is_open = True
            
    def close_serial(self):
        if self.is_open:
            self.ser.close()
            self.is_open = False
            print("Serial closed")            
            
    
    def read_serial(self, timeout = 30):
        timeout = time.time() + timeout
        
        data = ""
        stream_started = False
        while True:
            
            if time.time()>timeout:
                if self.is_open:
                    self.close_serial()
                raise Exception('Timeout in reading the data!')
                
            length = self.ser.inWaiting()
            
            if not stream_started:
                if length > 0:
                    stream_started = True
            else:
                if length == 0:
                    return data
                data = data + str(self.ser.read(length))
    
            time.sleep(0.5)
            
    def move_directly(self, command):
        self.is_homed = False
        if not self.is_open:
            self.open_serial()
        self.send_command(command)
        
        self.close_serial()
        
        print(f'Move done')

    def send_command(self, command):
        if not self.is_open:
            print('Serial is not open!')
            return ""
        
        self.ser.write(command.encode())
        
        response = self.read_serial()
        
        return response
    
    def get_position(self):
        return self.current_position


    
    def move_out(self):
        if not self.is_homed:
            print('Please home the motor')
            return
        
        self.move_to(-self.step_size)
        print("Attenuator is out")
      
        
        
    def move_in(self):
        if not self.is_homed:
            print('Please home the motor')
            return
        
        self.move_to(self.travel_range+self.step_size)
        print("Attenuator is in")    


        
    def move_to(self,position):
        if not self.is_homed:
            print('Please home the motor')
            return
        
        if self.current_position is None:
            raise Exception("Current position unknown")
            
        if position>self.travel_range:
            position = self.travel_range
        elif position < 0:
            position = 0
            
        distance = position - self.current_position
        
        steps = distance/self.step_size
        
        if steps!=0:
            steps = math.ceil(abs(steps)) * distance/abs(distance)
        
        self.move(steps)
    
    
    
    def move(self, steps):
        if not self.is_homed:
            print('Please home the motor')
            return
        
        steps = int(steps)
        
        print('Moving')
        
        
        
        if not self.is_open:
            self.open_serial()
            
        if abs(steps) * self.step_size > self.travel_range:
            steps = int (steps / abs(steps) * self.travel_range / self.step_size)
            
        if steps > 0:
            command = '-'
        elif steps < 0:
            command = '+'
        else:
            print('No move!')
            return
        
        command = command * abs(steps)
        
        response = self.send_command(command)
        
        end_position = self.current_position + steps*self.step_size
        
        if end_position > self.travel_range:
            self.current_position = self.travel_range
        elif end_position < 0:
            self.current_position = 0
        else:
            self.current_position = end_position
        

        self.close_serial()
        
        print(f'Move done to {self.current_position}')
        
        
    
    def home(self):
        print('Homing')
        
        
        
        if not self.is_open:
            self.open_serial()
                
        self.send_command('+++++++++++')
            
        response = self.send_command('h')
        
        self.close_serial()
        
        if not "Home position found:" in response:

            raise Exception('Homing error!')
        
        match = re.search('Home (\d+)',response)
        
        if match:
            home_position = int(match.group(1))
            self.home_position = home_position
            self.current_position = home_position
            
            print (f'Homed to {home_position}')
        
        self.is_homed = True
        
        
if __name__ == "__main__":
    
    attenuator = attenuator_control(4)
    while True:
        command = input("Type command (type help for help):")
        
        if command.lower() == 'help':
            print('List of commands (not case sensitive):')
            print('i or in for ND in')
            print('o or out for ND out')
            print('h or home to home')
            print('+ to shift out by steps')
            print('- to shift in by steps')
            print('exit to exit')
            print('integer position (e.g. 100) to move to position')
        elif command.lower() == 'i' or command.lower() == 'in':
            attenuator.move_in()
        elif command.lower() == 'o' or command.lower() == 'out':
            attenuator.move_out()
        elif command.lower() == 'h' or command.lower() == 'home':
            attenuator.home()
        elif command.lower() == 'exit':
            break
        elif command.lower().isdigit():      
            try :
                position = int(command)
                attenuator.move_to(position)
            except:
                print("position should be integer")
        elif '-' in command.lower() or '+' in command.lower():
            attenuator.move_directly(command)
        else:
            print("unknown command")
   
        
            