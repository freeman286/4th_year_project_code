import ms5837
import time

sensor = ms5837.MS5837_02BA() # Default I2C bus is 1 (Raspberry Pi 3)
#sensor = ms5837.MS5837_30BA(0) # Specify I2C bus
#sensor = ms5837.MS5837_02BA()
#sensor = ms5837.MS5837_02BA(0)
#sensor = ms5837.MS5837(model=ms5837.MS5837_MODEL_30BA, bus=0) # Specify model and bus

# We must initialize the sensor before reading it
if not sensor.init():
    print("Sensor could not be initialized")
    exit(1)

# Spew readings
while True:
    if sensor.read():
        print("MSL Relative Altitude: %.2f m Pressure: %.2f atm") % (sensor.altitude(), sensor.pressure(ms5837.UNITS_atm))
    else:
        print("Sensor read failed!")
        exit(1)
