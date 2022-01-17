import smbus
import time
import numpy as np
from timeit import default_timer as timer

def twos_comp(val, bits):
    """compute the 2's complement of int value val"""
    if (val & (1 << (bits - 1))) != 0: # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << bits)        # compute negative value
    return val  

# Get I2C bus
bus = smbus.SMBus(1)

bus.write_byte(0x40, 0x06) # Reset
bus.write_byte(0x40, 0x08) # START/SYNC

bus.write_i2c_block_data(0x40, 0x44, [0xD8]) # Set Register

bus.write_byte(0x40, 0x08) # START/SYNC

t = timer()

buffer = [0] * 2000 # A buffer of previous values
i = 0 # Index of buffer to write to

while (True) :        
    unsigned_value = int.from_bytes(bus.read_i2c_block_data(0x40,0x10,3), byteorder='big')
    
    signed_value = twos_comp(unsigned_value, 24)
    
    buffer[i] = signed_value
    
    i = (i + 1) % 2000
    
    time.sleep(max(0, 1/2000 - (timer() - t)))
    
    t = timer()