import RTIMU
import math

class BNO055:
    def __init__(self):
        self.settings = RTIMU.Settings("RTIMULib")
        self.imu = RTIMU.RTIMU(self.settings)
        self.imu.IMUInit()

        # Set the fusion mode to NDOF
        self.imu.setSlerpPower(0.02)
        self.imu.setAccelEnable(True)
        self.imu.setGyroEnable(True)
        self.imu.setCompassEnable(True)

        self.yaw = 0.0

        self.__degree_turn = 0

    def read_yaw(self):
        if self.imu.IMURead():
            data = self.imu.getIMUData()
            self.yaw =  math.degrees(data["fusionPose"][2])

        return self.yaw

    def __call__(self, initial_yaw):
        # Quay vật thể và tính toán degree turn
        current_yaw = self.read_yaw()
        self.__degree_turn = current_yaw - initial_yaw

        if self.__degree_turn < -180:
            self.__degree_turn += 360
        elif self.__degree_turn > 180:
            self.__degree_turn -= 360

        self.__degree_turn = abs(self.__degree_turn)
        # print("Degree turn: ", self.__degree_turn)

        return self.__degree_turn
            
if __name__ == "__main__":
    my_imu  = BNO055()
    init = True
    signal = 'left'

    initial_yaw = my_imu.read_yaw()
    degree_turn = 0
    angle_turn = 90

    while True:        
        while degree_turn < angle_turn:
            degree_turn = my_imu(initial_yaw)
            print(degree_turn)
        
        print("Done")