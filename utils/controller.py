import time
import numpy as np
import cv2

pre_t = time.time()
error_arr = np.zeros(5)


    
class Controller:
    def __init__(self, kp, kd):
        self.__kp = kp
        self.__kd = kd
        self.__ONE_LANEWIGHT = 65     # Độ rộng đường (pixel)
        self.__TWO_LANEWIGHT = self.__ONE_LANEWIGHT * 2 + 20
        self.width = 65

        self.mode = 0
        self.__turn_mode = 'None'

        self.__error = 0

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
        print(f"width: {self.width}")

        if 20 <= self.width <= self.__ONE_LANEWIGHT:
            print('1111')
            pass
        elif self.__ONE_LANEWIGHT < self.width <= self.__TWO_LANEWIGHT:
            print('2222')
            center = int((center + self.maxLane) / 2)
        else:
            print('3333')
            center = int(self.maxLane - self.__ONE_LANEWIGHT/2)

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

    def __linear(self, error):
        return int(-0.15*abs(error) + 25)

    def __call__(self, frame, sendBack_speed, height, signal, area):
        self.signal = signal
        ####### Angle Processing #######
        self.error = self.findingLane(frame, height)
        self.__sendBack_angle = -self.__PID(self.error)
        print(f"angle: {self.__sendBack_angle}")
        ####### Speed Processing #######
        self.__sendBack_speed =  sendBack_speed # self.__linear(self.error)

        #### Traffic Sign Processing ####
        # if 2000 <= area <= 4000:
        #     # print(signal, area)
        #     if signal == 'camtrai':
        #         self.__turn_mode = 'right'
        #     elif signal == 'camphai':
        #         self.__turn_mode = 'left'
        #     elif signal == 'phai':
        #         self.__turn_mode = 'right'
        #     elif signal == 'trai':
        #         self.__turn_mode = 'left'
        #     elif signal == 'thang':
        #         self.__turn_mode = 'straight'


        return self.__sendBack_angle, self.__sendBack_speed