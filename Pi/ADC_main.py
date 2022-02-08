#!/usr/bin/python
# -*- coding:utf-8 -*-


import time
import ADS1256
import RPi.GPIO as GPIO
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

i = 0

N = 1000
buffer1 = np.zeros(N)
buffer2 = np.zeros(N)


#try:
ADC = ADS1256.ADS1256()
ADC.ADS1256_init()
ADC.ADS1256_ConfigADC(ADS1256.ADS1256_GAIN_E['ADS1256_GAIN_1'] ,ADS1256.ADS1256_DRATE_E['ADS1256_15000SPS'])
ADC.ADS1256_SetDiffChannal(0)
t = timer()
ADC.ADS1256_StartContinuousADC()
while(1):
    ADC_value = ADC.ADS1256_Read_ADC_Data_Continuous()
        #print(ADC_value)
    x = timer()-t
    avgX=(avgX+x)/2
    t = timer()
    i=(i+1)%N
    buffer1[i] = ADC_value
    
    if (i==N-1):
        buffer2 = buffer1
    
    
T = 1/15000
freq = np.linspace(-1.0/(2.0*T), 1.0/(2.0*T), N)
fft = np.fft.fft(np.multiply(buffer2-np.mean(buffer2), np.hamming(N)))/N
plt.plot(freq, abs(fft))
plt.show()


