import time
import board
import busio
import adafruit_mpu6050

# Create I2C bus
i2c = busio.I2C(board.SCL, board.SDA)

# Create sensor object
mpu = adafruit_mpu6050.MPU6050(i2c)

while True:
    print("Acceleration (m/s^2):", mpu.acceleration)
    print("Gyro (rad/s):", mpu.gyro)
    print("Temperature (°C):", mpu.temperature)
    print("-" * 40)
    time.sleep(0.5)
