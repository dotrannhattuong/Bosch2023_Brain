import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np


vs = cv2.VideoCapture("./videos/record_2.avi")
count=0 
while 1:
	t =time.time()
	ret, frame = vs.read()
	h,w,_ = frame.shape
	frame = frame[400:635,int(w/2):int(w-(w/2-1000)),:]
	cv2.imshow('tuong',frame)
	h1,w1,_ = frame.shape
	count +=1

	if ret and count >2:
		count = 0    
		
		####################birdview
		tl = [0, 0]
		tr = [w1, 0]
		br = [w1, h1]
		bl = [0, h1]
		corner_points_array = np.float32([tl,tr,br,bl])
		h,w,c=frame.shape
		width = 2280
		height = 1640
		imgTl = [0,0]
		imgTr = [width,0]
		imgBr = [0.7*width,height]
		imgBl = [0.583*width,height]
		img_params = np.float32([imgTl,imgTr,imgBr,imgBl])

		matrix = cv2.getPerspectiveTransform(corner_points_array,img_params)
		img_transformed = cv2.warpPerspective(frame,matrix,(width,height))
		#####################

		img_transformed=img_transformed[int(0.46*height):height,int(0.46*width):int(0.83*width),:]
		ht,wt,c=img_transformed.shape
        
		#lọc màu trắng dựa vào RGB
		lower_white = np.array([200,200,200], dtype=np.uint8)
		upper_white = np.array([255,255,255], dtype=np.uint8)

		mask = cv2.inRange(img_transformed, lower_white, upper_white)

		# lọc nhiễu tường , gom vạch đi đường
		dilate = cv2.dilate(mask, None, iterations=9)
		mask = cv2.erode(dilate, None, iterations=10)
        
		mask1 = mask.copy()
		sampling = mask1[int(ht*0.89):int(0.98*ht),:]
		contours, _ = cv2.findContours(image=sampling, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
		normal = []
		a = np.zeros_like(sampling)
		for cnt in contours:
				x,y,wr,hr = cv2.boundingRect(cnt)
				area2 = cv2.contourArea(cnt)
				if area2 < 1000 and area2 >= 180: #đường bình thường
					normal.append(int(x+wr/2))
					cv2.circle(img_transformed, (int(x+wr/2),int(ht*0.935)), 5, (250,0,250), 15)
				elif area2 < 180: #nét đứt
					cv2.circle(a, (int(x+wr/2),int(y+hr/2)), 5, (250,0,250), 45) #gom các loại nét đứt 
				else:  #đường độ rộng lớn
					cv2.circle(img_transformed, (int(x),int(ht*0.935)), 5, (250,0,250), 15)
					cv2.circle(img_transformed, (int(x+wr),int(ht*0.935)), 5, (250,0,250), 15)
					normal.append(int(x+wr))
					normal.append(int(x))
		contours2, _ = cv2.findContours(image=a, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
		for cnt in contours2:
				x,y,wrs,hrs = cv2.boundingRect(cnt)
				area3 = cv2.contourArea(cnt)
				if area3 > 3000: # nét đứt gần
					normal.append(int(x+wrs/2))
					cv2.circle(img_transformed, (int(x+wrs/2),int(ht*0.935)), 5, (0,0,250), 15)
		normal = sorted(normal, reverse=True)[:2]
		b = None
		
		if len(normal) == 2: 
			if abs(normal[0]-normal[1])>200: # vạch đi đường
				b =  int(int((normal[0]+normal[1])/2) + abs(normal[0]-normal[1])/4)
			else:
				b = int((normal[0]+normal[1])/2)
		elif len(normal)  == 1: #một làn
			b = normal[0] - 50
		if b is not None:
			cv2.circle(img_transformed, (b,int(ht*0.935)), 5, (250,0,0), 15)
		cv2.imshow('mask',mask)
		cv2.imshow('res',sampling)
		plt.imshow(img_transformed)
		cv2.imshow('ss',img_transformed)
		print(time.time()-t)
	if cv2.waitKey(1) & 0xff == 27:
		cv2.destroyAllWindows()
		break 
vs.release()
cv2.destroyAllWindows() 