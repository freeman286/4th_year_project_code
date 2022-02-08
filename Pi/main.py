#!/usr/bin/python
# -*- coding:utf-8 -*-


import time
from datetime import datetime
import ADS1256
import RPi.GPIO as GPIO
import os
import numpy as np
from timeit import default_timer as timer
from modules.sensors import *

time.sleep(60)

write_path = os.getcwd() + '/data/data'+datetime.now().strftime("%Y.%m.%d.%H.%M.%S")+'.csv'
write_file = open(write_path, "w")

i = 0

#buffer parameters
N = 1000
buffer = np.zeros(N)

#fft parameters
T = 1/15000
freq = np.linspace(-1.0/(2.0*T), 1.0/(2.0*T), N)
scan_freq = 3500
scan_index = np.abs(freq-scan_freq).argmin()

#detection parameters
threshold = 30
alpha = 0.4
detection_level = -1000 #this starts very low while we wait for erroneous data to die down


angles = [0,0,0]

#try:
ADC = ADS1256.ADS1256()
ADC.ADS1256_init()
ADC.ADS1256_ConfigADC(ADS1256.ADS1256_GAIN_E['ADS1256_GAIN_1'] ,ADS1256.ADS1256_DRATE_E['ADS1256_15000SPS'])
ADC.ADS1256_SetDiffChannal(0)
t = timer()
ADC.ADS1256_StartContinuousADC()

while(1):
    ADC_value = ADC.ADS1256_Read_ADC_Data_Continuous()
    t = timer()
    i=(i+1)%N
    buffer[i] = ADC_value
        
    if (i==N-1):
        angles = read_angles(angles)
            
        fft = np.fft.fft(np.multiply(buffer-np.mean(buffer), np.hamming(N)))/N
            
        detection_level = alpha*abs(fft[scan_index])+(1-alpha)*detection_level   
                
        if (detection_level>threshold):
            print("detected")
            depth = read_depth()
            write_file.write(', '.join(map('{0:.2f}'.format, angles)) + ', {0:.4f}, '.format(depth) + datetime.now().strftime("%H.%M.%S") + '\n')
            write_file.flush()
            time.sleep(10)
            
