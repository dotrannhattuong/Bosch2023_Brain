import cv2
from utils.ser import Communication
from utils.controller import Controller
from utils.my_unet import Unet
from utils.my_realsense import RealSense
from utils.image_processing import img_processing
import numpy as np
import time

def main():
    # --------- Segment ---------#
    segment = Unet(weights='./weights/unet/unet_pytorch_8x16_pretrain.onnx')
    kernel = np.ones((5,5),np.uint8)

    # --------- Communication ---------#
    stm32 = Communication(port='/dev/ttyACM0')

    # --------- Controller ---------#
    Control = Controller(0.4, 0.05)
    
    # --------- RealSense ---------# 
    realsense = RealSense()

    # --------- Write Video ---------# 
    # result = cv2.VideoWriter('videochoLamLe.avi', 
    #                         cv2.VideoWriter_fourcc(*'MJPG'),
    #                         10, (320, 120))

    time.sleep(2)
    
    while True:
        try:   
            color_image, bg_removed, depth_colormap, images = realsense()

            # ------------------ Predict ------------------#
            # --------- Segment ---------#
            pred = segment(color_image)

            cv2.imshow('input', cv2.resize(color_image, (160, 80)))
            cv2.imshow('pred', pred)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                realsense.pipeline.stop()
                break
            
            # write_frame = np.hstack((cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR), cv2.resize(color_image[160:, :, :], (160, 120))))
            # result.write(write_frame)
            # result.write(color_image)

            # ------------------ Workspace ------------------#
            # --------- Controller ---------#
            sendBack_angle, sendBack_speed = Control(pred, sendBack_speed=100, height=45, signal='straight', area=10)
            
            # --------- Send Data ---------#
            stm32(speed=sendBack_speed,angle=sendBack_angle)

        except Exception as e:
            print(e)
            # result.release()
            continue
        

if __name__ == "__main__":
    main()