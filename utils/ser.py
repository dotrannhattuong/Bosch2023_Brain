import serial
import numpy as np

class Communication:
    def __init__(self, port="/dev/ttyACM1", baudrate=115200):
        self.stm32 = serial.Serial(port, baudrate, timeout=0.1)
    
    def __call__(self, speed=100, angle=0):
        ############ Preprocessng ############
        speed = self.speed_limit(speed)
        angle = self.angle_limit(angle)

        # print(speed)
        ############ Receive DATA ############
        # data = ser.readline().decode('utf-8')
        # print(f"data: {ser.readline().decode('utf-8')}")

        ############ SEND DATA ############
        ###### Encode ######
        send_0 = (np.uint16(speed) & 255).astype(np.uint8)
        send_1 = (np.uint16(speed) >> 8).astype(np.uint8)
        send_2 = np.uint8(angle)

        send_data = [send_0, send_1, send_2, 25]
        self.stm32.write(send_data)

        ###### Decode - DEBUG ######
        # print(f"Speed: {np.int16(send_1<<8)|send_0}")
        # print(f"Angle: {np.int8(send_2)}")

    @staticmethod
    def speed_limit(speed):
        return np.clip(speed, -500, 500)

    @staticmethod
    def angle_limit(angle):
        return np.clip(angle, -25, 25)


if __name__ == "__main__":
    stm32 = Communication()

    while True:
        stm32(speed=0, angle=0)
    