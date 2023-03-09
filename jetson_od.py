# -*- coding: utf-8 -*-
import socket
import cv2
import numpy as np
import time
import threading
import copy

class Jet_OD:
    def __init__(self):
        # receive socket
        self.sock_rec = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock_rec.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock_rec.bind(('10.0.0.20', 8001))
        self.sock_rec.listen()
        self.conn_rec, self.addr_rec = self.sock_rec.accept()

        # send socket
        self._sock_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock_send.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock_send.bind(('10.0.0.20', 1234))
        self._sock_send.listen()
        self.conn_send, self.addr_send = self._sock_send.accept()

        # var
        self.img = np.zeros((480, 640))

    def _process(self):
        while True:
            # OD task
            OD_result = 'abcxyz'
            # Send result to server
            self.conn_send.sendall(f'{OD_result}'.encode())
    
    def _rec(self):
        while True:
            # Receive Steam --> self.img
            header_str = self.conn_rec.recv(7)
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
                print("header - len()", header - len(img_bytes))
                chunk = self.conn_rec.recv(header - len(img_bytes)) 
                if not chunk:
                    print(f'break do  chunk rỗng ')
                    break
                img_bytes += chunk 
            # print('kich thươc img nhận: ', len(img_bytes))
            self.img = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            # cv2.imwrite("hstack.png", self.img)
            
