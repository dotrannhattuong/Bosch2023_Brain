# -*- coding: utf-8 -*-
import socket
import cv2
import numpy as np
import time
import threading
from utils.my_unet import Unet
from utils.controller import Controller
from utils.my_unet import Unet
from utils.my_realsense import RealSense
from utils.image_processing import img_processing, lamle

class Jet_Seg:
    def __init__(self):
        # send socket
        self.sock_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                self.sock_send.connect(('10.0.0.30', 8001))
                print('Jet_Seg_send connected')
                break
            except Exception as e:
                print("Try to connect to port 8001.")
                time.sleep(0.5)
                continue
        
        print('Jet_Seg_send connected')

        # receive socket
        self.sock_rec = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                self.sock_rec.connect(('10.0.0.30', 1234))
                print('Jet_Seg_rec connected')
                break
            except Exception as e:
                print("Try to connect to port 1234.")
                time.sleep(0.5)
                continue

        # var
        self.receive_od = None
        self.clss = 10
        self.c1 = 0
        self.c2 = 0
        self.distance = 1000
        self.segment = Unet(weights='weights/unet/unet_pytorch_8x16_pretrain.onnx')
        self.Control = Controller(0.25, 0.05)
        self.realsense = RealSense()
        self.color_image = np.zeros((480, 640))
        self.depth_frame = np.zeros((480, 640))
        self.depth_colormap = np.zeros((480, 640))
        time.sleep(2)
        self.count_send = 0

    def _send(self):
        while True:
            try:
                # Stream Cam
                # self.color_image = cv2.cvtColor(cv2.imread("2376.png"), cv2.COLOR_BGR2RGB)
                # self.color_image, self.depth_frame, self.depth_colormap, self.frames = self.realsense()
                self.color_image, self.bg_removed, self.depth_frame, self.images = self.realsense()
                # send_img = np.hstack((self.bg_removed, self.depth_colormap))

                # Send Image
                img_bytes = cv2.imencode('.png', self.images)[1].tobytes() # print('gửi header kich thước ', len(img_bytes))
                header = f'{len(img_bytes)}'.encode()
                self.sock_send.sendall(header) # print('gửi ảnh ', len(img_bytes))
                self.sock_send.sendall(img_bytes)
                self.count_send += 1
            except Exception as e:
                print(e)
                continue

    def _rec(self):
        while True:
            # Receive OD 
            try:
                self.receive_od = self.sock_rec.recv(3000000).decode()
                self.clss, self.c1, self.c2 = self.receive_od.split(' ')[0], self.receive_od.split(' ')[1], self.receive_od.split(' ')[2]
                self.distance = self.depth_frame.get_distance(int(self.c1),int(self.c2))
                print("receiving..", self.clss, self.distance, self.c1, self.c2)
            except Exception as e:
                print(e)
                continue
    def _process(self):
        while True:
            try:   
                s = time.time()
                # Process Control + Segment + OD
                pred = self.segment(self.color_image)
                # pred = lamle(pred)
                
                cv2.imshow('input', cv2.resize(self.color_image, (160, 80)))
                cv2.imshow('pred', pred)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    realsense.pipeline.stop()
                    break

                # OD
                # print('process', self.receive_od)
                # time.sleep(0.5)

                # Control
                self.Control(pred, sendBack_speed=100, height=50, class_id=int(self.clss), distance=float(self.distance))

                # print('time', time.time() - s)
            except Exception as e:
                print(e)
                continue
        self.realsense.pipeline.stop()