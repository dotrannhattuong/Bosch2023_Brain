import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np


vs = cv2.VideoCapture("./videos/record_1.avi")
count=0 
init = True

while 1:
	t =time.time()
	ret, frame_or = vs.read()
	h,w,_ = frame_or.shape

	frame = frame_or[300:600,:int(w/2),:]
	frame1 = frame_or[300:600,int(w/2):w,:]
	h,w,_ = frame.shape

	image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	if init:
		calib = image_gray[:,int(w/2)] ##w/2 không có vật cản
		init= False

	height_sample = 10 #trade-off với resolution vật cản và tốc độ thuật toán
	for i in range(int(h/height_sample)):
		sample = image_gray[i*height_sample:i*height_sample+height_sample,:]
		sample = np.mean(sample, axis=0)
		error = abs(sample - calib[i*height_sample+int(height_sample/2)]) #khác biệt so với đường calib
		binary = np.where(error < 10, 255, 0) #10 độ là độ chênh lệch, lọc các bề mặt nhô cao so hơn với đường
		for j in range(len(binary)):
			if  binary[j] == 0:
				cv2.circle(image_gray, (int(j),int(i*height_sample+int(height_sample/2))), 2, (250,0,250), 3)

	plt.imshow(image_gray)
	# cv2.imshow('ss',crop_bw)
	cv2.imshow('â',image_gray)
	cv2.imshow('âv21321v',frame)
	cv2.imshow('âvv',frame1)

	if cv2.waitKey(1) & 0xff == 27:
		cv2.destroyAllWindows()
		break 
vs.release()
cv2.destroyAllWindows() 