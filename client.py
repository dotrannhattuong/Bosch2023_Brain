import cv2
from utils.ser import Communication
from utils.controller import Controller
from utils.my_unet import Unet
from utils.my_realsense import RealSense

def main():
    # --------- Segment ---------#
    segment = Unet(weights='bosch_3c_sof.onnx')

    # --------- Communication ---------#
    # stm32 = Communication(port='/dev/ttyACM0')

    # --------- Controller ---------#
    # Control = Controller(1, 0.05)

    # --------- RealSense ---------# 
    realsense = RealSense()

    # --------- Write Video ---------# 
    # result = cv2.VideoWriter('filename.avi', 
    #                         cv2.VideoWriter_fourcc(*'MJPG'),
    #                         10, (160, 80))

    try:
        while True:
            color_image, depth_frame, depth_colormap = realsense()

            # ------------------ Predict ------------------#
            # --------- Segment ---------#
            pred = segment(color_image)

            cv2.imshow('input', color_image)
            cv2.imshow('pred', pred)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # result.write(cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB))

            # ------------------ Workspace ------------------#
            # --------- Controller ---------#
            # sendBack_angle, sendBack_speed = Control(pred, sendBack_speed=120, height=15, signal='straight', area=10)
            
            # --------- Send Data ---------#
            # stm32(speed=sendBack_speed,angle=sendBack_angle)

    finally:
        realsense.pipeline.stop()
        # result.release()

if __name__ == "__main__":
    main()