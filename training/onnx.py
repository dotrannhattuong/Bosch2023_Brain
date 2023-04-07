# -*- coding: utf-8 -*-
import math


import os
import time
import argparse
import numpy as np
import onnxruntime
import sys
import time
import cv2
import time


def road_lines(color_image, session, inputname):
    image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    image = image[200:, :, :]
    image = cv2.resize(image, (160, 80))
    image = image/255
    image = np.array(image, dtype=np.float32) # H x W x 3
    # image = np.swapaxes(image, -1, 0) # 3 x H x W
    # image = np.swapaxes(image, -1, 1) # 3 x W x H
    image = image.transpose((2, 0, 1))
    image = image[None, :, :, :]
    prediction = session.run(None, {inputname: image})
    prediction = np.squeeze(prediction)
    prediction = np.where(prediction > 0.1, 255, 0)
    prediction = prediction.astype(np.uint8)
    return prediction

if __name__=="__main__":
    img = cv2.imread('2908.png')

    session_lane = onnxruntime.InferenceSession('unet_pytorch_8x16.onnx', None, providers=['CPUExecutionProvider'])
    input_name_lane = session_lane.get_inputs()[0].name

    import time
    from glob import glob

    img_paths = glob("segment/imgs/*.png")    
    for i, ip in img_paths:
        img = cv2.imread('2908.png')
        pred = road_lines(img, session=session_lane,inputname=input_name_lane)    #hàm detect làn đường trả về ảnh đã detect, góc lái, với điểm center dự đoán      
        cv2.imwrite(f'img.png', img)
        cv2.waitKey(0)
