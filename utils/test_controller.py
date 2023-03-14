import time
import numpy as np
import cv2
from utils.my_bno055 import BNO055
from utils.ser import Communication

if __name__ == "__main__":
    stm32 = Communication(port='/dev/ttyACM0')
    my_imu = BNO055()


    time_straight = 0.9
    setpoint_turn = 88
    sendBack_speed = 100
    sendBack_angle = -11
    test = True

    i = 0

    while True:
        if test == True:
            ### Stage 1 ###
            start_time = time.time()
            print("Stage 1")
            while(time.time() - start_time < time_straight):
                
                stm32(speed=sendBack_speed,angle=0)

            ### Stage 2 ###
            initial_yaw = my_imu.read_yaw()
            degree_turn = 0

            print("Stage 2")
            while degree_turn < setpoint_turn:
                degree_turn = my_imu(initial_yaw)
                stm32(speed=sendBack_speed,angle=sendBack_angle)

            i += 1

            if i == 1:
                test = False

        stm32(speed=0,angle=0) 
