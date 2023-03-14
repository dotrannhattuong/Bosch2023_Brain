import time
import numpy as np
import cv2
from utils.my_bno055 import BNO055
from utils.ser import Communication

pre_t = time.time()
error_arr = np.zeros(5)
    
class Controller:
    def __init__(self, kp, kd):
        self.__kp = kp
        self.__kd = kd
        self.__ONE_LANEWIGHT = 70     # Độ rộng đường (pixel)
        self.__TWO_LANEWIGHT = self.__ONE_LANEWIGHT * 2 + 20
        self.width = self.__ONE_LANEWIGHT

        self.mode = 0
        self.__turn_mode = 'None'

        self.__error = 0

        self.my_imu = BNO055()
        self.status = True

        self.__names = ['crosswalk_sign', 'highway_entrance_sign', 'highway_exit_sign', \
        'no_entry_road_sign', 'one_way_sign', 'parking_sign', 'priority_sign', 'round_about_sign', \
        'stop_sign', 'traffic_light', 'none', 'red', 'yellow', 'green']

        self.stm32 = Communication(port='/dev/ttyACM0')

        self.signs = 'straight'
        
        self.flag = 0

    def findingLane(self, frame, height):
        arr_normal = []
        lineRow = frame[height, :]
        for x, y in enumerate(lineRow):
            if y == 255:
                arr_normal.append(x)
        if not arr_normal:

            return 20
        
        self.minLane = min(arr_normal)
        self.maxLane = max(arr_normal)

        center = int((self.minLane + self.maxLane) / 2)
        
        #### Safe Mode ####
        self.width=self.maxLane - self.minLane
        # print(f"width: {self.width}")

        if 20 <= self.width <= self.__ONE_LANEWIGHT:
            pass
        elif self.__ONE_LANEWIGHT < self.width <= self.__TWO_LANEWIGHT:
            center = int((center + self.maxLane) / 2)
        else:
            center = int(self.maxLane - self.__ONE_LANEWIGHT/2 - 10)

        #### Cua sớm ####
        # if (0 < self.width < self.__LANEWIGHT):
        #     if (center < int(frame.shape[1]/2)):
        #         center -= self.__LANEWIGHT - self.width
        #     else :
        #         center += self.__LANEWIGHT - self.width

        #### Error ####
        error = frame.shape[1]//2 - center
        self.__error = error

        return error

    def __PID(self, error):
        global pre_t
        # global error_arr
        error_arr[1:] = error_arr[0:-1]
        error_arr[0] = error
        P = error*self.__kp
        delta_t = time.time() - pre_t
        pre_t = time.time()
        D = (error-error_arr[1])/delta_t*self.__kd
        angle = P + D
        
        if angle >= 25:
            angle = 25
        elif angle <= -25:
            angle = -25

        return int(angle)

    def __timer_intersection(self, time_straight, sendBack_speed, sendBack_angle, setpoint_turn):
        ### Stage 1 ###
        start_time = time.time()
        while(time.time() - start_time < time_straight):
            self.stm32(speed=sendBack_speed,angle=0)

        ### Stage 2 ###
        initial_yaw = self.my_imu.read_yaw()
        degree_turn = 0

        while degree_turn < setpoint_turn:
            degree_turn = self.my_imu(initial_yaw)
            self.stm32(speed=sendBack_speed,angle=sendBack_angle)

        ##### Restart Status #####
        self.status = True
        self.signs = 'straight'
        self.distance = 100

    def __timer_stop(self, timer_stop, timer_straight, speed_straight):
        ### Stage 1.1 ###
        start_time = time.time()
        while(time.time() - start_time < 0.2):
            self.stm32(speed=speed_straight,angle=0)

        ### Stage 1 ###
        start_time = time.time()
        while(time.time() - start_time < timer_stop):
            self.stm32(speed=0,angle=0)

        ### Stage 2 ###
        start_time = time.time()
        while(time.time() - start_time < timer_straight):
            self.stm32(speed=speed_straight,angle=0)

        ##### Restart Status #####
        self.status = True
        self.signs = 'straight'
        self.distance = 100

    def __timer_light(self):
        self.stm32(speed=0,angle=0)

    def __call__(self, frame, sendBack_speed, height, class_id, distance):
        signs = self.__names[class_id]

        # if self.status == True:
        # self.distance = distance

        # if self.distance <= 0.8 and self.status == True:
        #     self.signs = signs
        #     self.status = False

        # print('*'*60)
        print(signs, '-', distance)
        # print('*'*60)

        ####### Angle Processing #######
        self.error = self.findingLane(frame, height)
        self.__sendBack_angle = -self.__PID(self.error)
        ####### Speed Processing #######
        self.__sendBack_speed =  sendBack_speed # self.__linear(self.error)

        #### Traffic Sign Processing ####
        if signs == 'green':
            self.flag = 0

        if 0.2 <= distance <= 0.70 and self.flag == 0:
            self.signs = signs

        if 0.2 <= distance <= 0.60:
            # self.signs = signs

            if self.signs == 'highway_entrance_sign':
                self.flag = 0
                self.__timer_intersection(time_straight=0.6, sendBack_speed=100, sendBack_angle=-15, setpoint_turn=88)
            elif self.signs == 'stop_sign':
                self.flag = 0
                self.__timer_stop(timer_stop=2, timer_straight=1.2, speed_straight=100)
            elif self.signs == 'green':
                self.flag = 0
                self.__timer_intersection(time_straight=1, sendBack_speed=90, sendBack_angle=25, setpoint_turn=85)
            elif self.signs == 'red':
                self.flag = 1
                self.__timer_light()
            elif self.signs == 'yellow':
                self.flag = 1
                self.__timer_light()
            elif self.signs == 'no_entry_road_sign':
                self.flag = 0
                self.__timer_intersection(time_straight=0.8, sendBack_speed=100, sendBack_angle=25, setpoint_turn=70)
                
        print(self.flag)
        # Send data to STM
        if self.flag == 0:
            self.stm32(speed=self.__sendBack_speed,angle=self.__sendBack_angle)