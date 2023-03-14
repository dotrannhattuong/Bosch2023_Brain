# -*- coding: utf-8 -*-
import socket
import cv2
import numpy as np
import time
import threading
import copy
from utils.detect_yolo import YOLO_DETECT

class Jet_OD:
    def __init__(self):
        self.detector = YOLO_DETECT(engine_path='./weights/yolo/best-v1-nms.trt', imgsz=(448,448))
        # YOLO
        print('ok')

        # receive socket
        self.sock_rec = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock_rec.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock_rec.bind(('10.0.0.30', 8001))
        self.sock_rec.listen()
        self.conn_rec, self.addr_rec = self.sock_rec.accept()

        # send socket
        self._sock_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock_send.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock_send.bind(('10.0.0.30', 1234))
        self._sock_send.listen()
        self.conn_send, self.addr_send = self._sock_send.accept()

        # var
        self.img = np.zeros((480, 640))
        self.bg_removed = np.zeros((480, 640, 3))
        self.depth_colormap = np.zeros((480, 640, 3))
        self.class_id = 0
        self.count_rec = 0
        self.center = (1000, 1000)
        self.score = 20
    def _process(self):
        while True:
            # OD task
            try:
                # boxes, scores, classes, self.class_id, self.distance, img_pred = self.detector(self.bg_removed, self.depth_colormap)
                _, self.score, _, self.class_id, img_pred, self.center, name, s, log = self.detector(self.bg_removed, self.depth_colormap, vis = True)
                
                # img_pred = self.detector.visualize(self.bg_removed, boxes, self.scores, classes)
                # cv2.imwrite('img_pred.png', img_pred)
                cv2.imshow('img_pred', cv2.resize(img_pred, (320, 160)))
                # Send result to server
                print(f'Signal: {name} | Center: {self.center} | Score: {self.score} | S: {s}', log)
                self.conn_send.sendall(f'{self.class_id} {self.center[0]} {self.center[1]} '.encode())
                # time.sleep(0.0001)
            except Exception as e:
                print(e)
                continue
    def _rec(self):
        while True:
            try:
                # Receive Steam --> self.img
                header_str = self.conn_rec.recv(6)
                # print("len header", len(header_str))
                header_str = header_str.decode()
                if(header_str == ''):
                    print('break do size_str là rỗng')
                    break
                else:
                    header = int(header_str) 
                # print(f'header: {header}')
                img_bytes = b''
                while len(img_bytes) < header:
                    # print("header - len()", header - len(img_bytes))
                    chunk = self.conn_rec.recv(header - len(img_bytes)) 
                    if not chunk:
                        print(f'break do  chunk rỗng ')
                        break
                    img_bytes += chunk 
                # print('kich thươc img nhận: ', len(img_bytes))
                self.img = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                self.bg_removed, self.depth_colormap = self.img[:, :640], self.img[:, 640:]
                self.count_rec += 1 # debug
                # time.sleep(0.0001)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    realsense.pipeline.stop()
                    break
            except Exception as e:
                print(e)
                continue