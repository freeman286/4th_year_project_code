import ms5837
import smbus
import time

bus = smbus.SMBus(1)

sensor = ms5837.MS5837_02BA() # Default I2C bus is 1 (Raspberry Pi 3)

time.sleep(0.1)

if not sensor.init():
    print("Sensor could not be initialized")
    exit(1)


def read_angles(previous_angles):
    data = [0, 0, 0, 0, 0, 0]
    raw_adc = angle = [0, 0, 0]
    
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
    
    # Spool wrap around    
    if (previous_angles[2] % 360 > 270 and angle[2] < 90):
        angle[2] += 360 * ((previous_angles[2] // 360) + 1)
    elif (previous_angles[2] % 360 < 90 and angle[2] > 270):
        angle[2] += 360 * ((previous_angles[2] // 360) - 1)
    else:
        angle[2] += 360 * (previous_angles[2] // 360)
        
    return angle



def read_depth():
    if sensor.read():
        return sensor.depth()
    else:
        print("Sensor read failed!")
        exit(1)