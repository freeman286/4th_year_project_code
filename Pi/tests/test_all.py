# Distributed with a free-will license.
# Use it any way you want, profit or free, provided it fits in the licenses of its associated works.
# A1332
# This code is designed to work with the A1332_I2CS I2C Mini Module available from ControlEverything.com.
# https://www.controleverything.com/content/Hall-Effect?sku=A1332_I2CS#tabs-0-product_tabset-2

import smbus
import time

# Get I2C bus
bus = smbus.SMBus(1)

# A1332 address, 0x0C(12)
# Read data back, 2 bytes
# raw_adc MSB, raw_adc LSB

data = [0, 0, 0, 0, 0, 0]
raw_adc = angle = [0, 0, 0]

while (0 in data) :
    
    data[0] = bus.read_byte(0x0C)
    data[1] = bus.read_byte(0x0C)
    
    data[2] = bus.read_byte(0x0D)
    data[3] = bus.read_byte(0x0D)
    
    data[4] = bus.read_byte(0x0D)
    data[5] = bus.read_byte(0x0D)
    
print("Receiving data")

time.sleep(0.5)

while (True) :
    
    data[0] = bus.read_byte(0x0C)
    data[1] = bus.read_byte(0x0C)
    
    data[2] = bus.read_byte(0x0D)
    data[3] = bus.read_byte(0x0D)
    
    data[4] = bus.read_byte(0x0D)
    data[5] = bus.read_byte(0x0D)

    for i in range(3) :
        # Convert the data to 12-bits
        raw_adc[i] = (data[2*i] & 0x0F) * 256.0 + data[2*i+1]
        angle[i] = (raw_adc[i] / 4096.0) * 360.0

    # Output data to screen
    print("Magnetic Angle : " + ", ".join(str(j) for j in angle))
    
    time.sleep(0.1)
