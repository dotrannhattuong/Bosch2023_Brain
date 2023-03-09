# -*- coding: utf-8 -*-
import socket
import cv2
import numpy as np
import time
import threading
from utils.my_unet import Unet
from utils.ser import Communication
from utils.controller import Controller
from utils.my_unet import Unet
from utils.my_realsense import RealSense
from utils.image_processing import img_processing

class Jet_Seg:
    def __init__(self):
        # send socket
        self.sock_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock_send.connect(('10.0.0.20', 8001))
        print('Jet_Seg_send connected')

        # receive socket
        self.sock_rec = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                self.sock_rec.connect(('10.0.0.20', 1234))
                print('Jet_Seg_rec connected')
                break
            except Exception as e:
                print("Try to connect to port 1234.")
                time.sleep(0.5)
                continue

        # var
        self.od = None
        self.segment = Unet(weights='bosch_sig_1c_16x12.onnx')
        self.stm32 = Communication(port='/dev/ttyACM0')
        self.Control = Controller(0.3, 0.05)
        self.realsense = RealSense()
        self.color_image = np.zeros((480, 640))
        self.depth_frame = np.zeros((480, 640))
        self.depth_colormap = np.zeros((480, 640))
        time.sleep(2)

    def _send(self):
        while True:
            # Stream Cam
            self.color_image, self.depth_frame, self.depth_colormap = self.realsense()
            send_img = np.hstack((self.color_image, self.depth_frame, self.depth_colormap))

            # Send Image
            img_bytes = cv2.imencode('.png', send_img)[1].tobytes() # print('gửi header kich thước ', len(img_bytes))
            header = f'{len(img_bytes)}'.encode()
            self.sock_send.sendall(header) # print('gửi ảnh ', len(img_bytes))
            self.sock_send.sendall(img_bytes)

    def _rec(self):
        while True:
            # Receive OD 
            self.od = self.sock_rec.recv(1024).decode()

    def _process(self):
        while True:
            try:   
                # Process Control + Segment + OD
                pred = self.segment(color_image)
                cv2.imshow('input', cv2.resize(self.color_image, (160, 80)))
                cv2.imshow('pred', pred)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    realsense.pipeline.stop()
                    break

                # OD
                print('process', self.od)
                time.sleep(0.5)

                # Control
                sendBack_angle, sendBack_speed = Control(pred, sendBack_speed=100, height=30, signal='straight', area=10)

                # Send data to STM
                self.stm32(speed=sendBack_speed,angle=sendBack_angle)
                
            except Exception as e:
                print(e)
                continue