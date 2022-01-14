import smbus
import time
import numpy as np

def twos_comp(val, bits):
    """compute the 2's complement of int value val"""
    if (val & (1 << (bits - 1))) != 0: # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << bits)        # compute negative value
    return val  

# Get I2C bus
bus = smbus.SMBus(1)

while (True) :
    bus.write_byte(0x40, 0x10)
    
    data1 = bus.read_byte(0x40)
    data2 = bus.read_byte(0x40)
    data3 = bus.read_byte(0x40)

    unsigned_value = int.from_bytes([data1,data2,data3], byteorder='big')
    signed_value = twos_comp(unsigned_value, 24)
            
            
    print("x"*int(abs(signed_value)/10))
    time.sleep(1/2000)
