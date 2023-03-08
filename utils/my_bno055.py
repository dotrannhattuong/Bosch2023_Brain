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

    def read_yaw(self):
        if self.imu.IMURead():
            data = self.imu.getIMUData()
            self.yaw =  math.degrees(data["fusionPose"][2])

        return self.yaw

if __name__ == "__main__":
    my_imu  = BNO055()

    while True:
        yaw_angle = my_imu.read_yaw()
        print("Yaw angle: {:.2f} degrees".format(yaw_angle))