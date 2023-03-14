import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy

def img_processing(img):
    img = img[160:, :, :]
    
    blur = cv2.medianBlur(img, 11)
    canny = cv2.Canny(blur, 150, 200)

    zeros_img = np.zeros_like(canny)
    points = np.array([[0, 320], [0, 150], [140, 0], [500, 0], [640, 150], [640, 320]])
    poly = cv2.fillPoly(zeros_img, pts=[points], color=(255, 255, 255))

    output = cv2.bitwise_and(canny, poly)
    output = cv2.resize(output, (160,80))
    return output

def lamle(mask_input):
    mask = np.zeros(mask_input.shape, np.uint8)
    dst_bw = cv2.dilate(mask_input, None, iterations=2)
    contours, _ = cv2.findContours(image=dst_bw, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    for con in contours:
        area = cv2.contourArea(con)
        if area>100:
            cv2.fillPoly(mask, pts =[con], color=(255,255,255))
    output = cv2.bitwise_and(mask_input, mask_input, mask=mask)
    return output, mask