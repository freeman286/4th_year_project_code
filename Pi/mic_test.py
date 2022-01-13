import smbus
import time

# Get I2C bus
bus = smbus.SMBus(1)

bus.write_word_data(0x40, 0x40, 0x08) # Start command

bus.write_word_data(0x40, 0x40, 0x44) # Set register 0 to 1

while (True) :
    data = bus.read_byte(0x40)

    volts = data / 4096000

    print(volts)
