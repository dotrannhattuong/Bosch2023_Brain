import time
import numpy as np
import cv2

pre_t = time.time()
error_arr = np.zeros(5)


    
class Controller:
    def __init__(self, kp, kd):
        self.__kp = kp
        self.__kd = kd
        self.__LANEWIGHT = 70            # Độ rộng đường (pixel)
        self.width = 70

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
            # arr_normal = [frame.shape[1] * 1 // 3, frame.shape[1] * 2 // 3]
            return 0

        self.minLane = min(arr_normal)
        self.maxLane = max(arr_normal)
        
        center = int((self.minLane + self.maxLane) / 2)
        
        #### Safe Mode ####
        self.width=self.maxLane-self.minLane
        print(self.width)

        #### Cua sớm ####
        if (0 < self.width < self.__LANEWIGHT):
            if (center < int(frame.shape[1]/2)):
                center -= self.__LANEWIGHT - self.width
            else :
                center += self.__LANEWIGHT - self.width

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
        
        if angle+5 >= 55:
            angle = 55
        elif angle+5 <= -45:
            angle = -45

        return int(angle)
    
    def __timer_intersection(self, time_turn, angle_turn, velo):
        Car.setSpeed_rad(velo)

        if self.width >= 120:
            # print('Timerrrrrrrrrrrrrrrrrr')
            Car.setAngle(angle_turn) #### Set Angle

            start_time = time.time()
            while(time.time() - start_time < time_turn):
                ### Safe Mode ###
                if Car.button == 3:
                    Car.setAngle(angle_turn)
                    Car.setSpeed_rad(velo)
                elif Car.button == 2:
                    Car.setAngle(0)
                    Car.setSpeed_rad(0)

            self.__turn_mode = 'None'
        else: 
            Car.setAngle(self.__sendBack_angle+5)
            Car.setSpeed_rad(self.__sendBack_speed)

    def __turnright(self):
        # print('Rightttttttttttttttttttttttt')
        self.__timer_intersection(time_turn=1, velo=25, angle_turn=-45)

    def __turnleft(self):
        # print('Leftttttttttttttttttttttt')
        self.__timer_intersection(time_turn=1, velo=25, angle_turn=45)
    
    def __straight(self):
        # print('Straightttttttttttttttttt')

        if self.width >= 110:
            # print('Timerrrrrrrrrrrrrrrrrr')
            Car.setAngle(5) #### Set Angle

            start_time = time.time()
            while(time.time() - start_time < 0.2):
                ### Safe Mode ###
                if Car.button == 3:
                    Car.setAngle(5)
                    Car.setSpeed_rad(25)
                elif Car.button == 2:
                    Car.setAngle(0)
                    Car.setSpeed_rad(0)

            self.__turn_mode = 'None'
        else:
            Car.setSpeed_rad(22)

            if 0 <= self.minLane <= 30 and 80 <= self.maxLane <= 135: # and self.width > 90:
                center = self.maxLane - self.__LANEWIGHT//2
                Car.setAngle(self.__PID(80 - center) + 5)
                # print('Change Max', self.minLane, self.maxLane)
            elif 35 <= self.minLane <= 80 and self.maxLane >= 129: # and self.width > 90:
                center = self.minLane + self.__LANEWIGHT//2
                Car.setAngle(self.__PID(80 - center))
                # print('Change Min', self.minLane, self.maxLane)
                
            else:
                Car.setAngle(self.__sendBack_angle+5)

        # self.__timer_intersection(time_turn=0.5, velo=28, angle_turn=6)

    def __linear(self, error):
        return int(-0.15*abs(error) + 25)

    def __call__(self, frame, sendBack_speed, height, signal, area):
        self.signal = signal
        ####### Angle Processing #######
        self.error = self.findingLane(frame, height)
        self.__sendBack_angle = -self.__PID(self.error)

        ####### Speed Processing #######
        self.__sendBack_speed =  sendBack_speed # self.__linear(self.error)

        ####### Show Info #######
        # self.__show(signal, area)

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

        ####### Start-Stop #######
        # print(sendBack_speed)

        return self.__sendBack_angle, self.__sendBack_speed