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