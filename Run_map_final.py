import socket       
import sys      
import time
import cv2
import numpy as np
import json
import base64
from scipy import stats
import cv2
import numpy as np

def myfunc(x):
  return slope * x + intercept

# Create a socket object 
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 

# Define the port on which you want to connect 
port = 54321                
  
# connect to the server on local computer 
s.connect(('127.0.0.1', port)) 

count = 0
angle = 10
speed = 100


def morphology(b_img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    b_img = cv2.morphologyEx(b_img, cv2.MORPH_OPEN, kernel, iterations=3)
    b_img = cv2.morphologyEx(b_img, cv2.MORPH_OPEN, kernel, iterations=1)
    b_img = cv2.dilate(b_img, kernel,iterations=4)
    return b_img

def remove_small_contours(image):
    try: 
        image_binary = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        contours = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        mask = cv2.drawContours(image_binary, [max(contours, key=cv2.contourArea)], -1, (255, 255, 255), -1)
        image_remove = cv2.bitwise_and(image, image, mask=mask)
        return image_remove
    except:
        return image

def control_speed(angle):
    if abs(angle) >= 0 and  abs(angle) <= 2:
        speed = 140
    elif abs(angle) > 2  and  abs(angle) <= 15:
        speed = 80
    elif abs(angle) > 15  and  abs(angle) <= 18:
        speed = 80
    else:
        speed = -1.5
    return speed
# def control_speed(angle):
#     if abs(angle) >= 0 and  abs(angle) <= 2:
#         speed = 80
#     elif abs(angle) > 2  and  abs(angle) <= 15:
#         speed = 60
#     elif abs(angle) > 15  and  abs(angle) <= 20:
#         speed = 40
#     else:
#         speed = -30
#     return speed

pre_t = time.time()
error_arr = np.zeros(5)
def PID(error, p, i, d): #0.43,0,0.02
    global pre_t
    # global error_arr
    error_arr[1:] = error_arr[0:-1]
    error_arr[0] = error
    P = error*p
    delta_t = time.time() - pre_t
    #print('DELAY: {:.6f}s'.format(delta_t))
    pre_t = time.time()
    D = (error-error_arr[1])/delta_t*d
    I = np.sum(error_arr)*delta_t*i
    angle = P + I + D
    if abs(angle)>25:
        angle = np.sign(angle)*25
    return int(angle)



if __name__ == "__main__":
    try:
        prev_frame_time = 0
        new_frame_time = 0
        message = bytes(f"{angle} {speed}", "utf-8")
        s.sendall(message)
        while True:
           
            # Recive data from server
            data = s.recv(100000000)
            
            # print(data)
            try:
                data_recv = json.loads(data)
            except:
                print("No received data")
                continue
            # Angle and speed recv from server
            current_angle = data_recv["Angle"]
            current_speed = data_recv["Speed"]
            # print("angle: ", current_angle)
            # print("SPEED: ", current_speed)
            # print("---------------------------------------")
            #Img data recv from server
            jpg_original = base64.b64decode(data_recv["Img"])
            jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
            image = cv2.imdecode(jpg_as_np, flags=1)
        #====================segment==============================
            img_show = image.copy()
            image = image[250:,:,:]
            img = cv2.GaussianBlur(image, (7,7), 0)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            b_img = cv2.inRange(hsv, (0,77,0), (179,255,140))
            b_img = cv2.threshold(b_img,100,255,cv2.THRESH_BINARY_INV)[1]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            b_img = cv2.morphologyEx(b_img, cv2.MORPH_OPEN, kernel, iterations=10)
            b_img = remove_small_contours(b_img)
            b_img = cv2.morphologyEx(b_img, cv2.MORPH_CLOSE, kernel, iterations=10)
            # print(b_img.shape)
            b_img[0,:] = np.zeros((1,640), dtype=int)
            # b_img[:,639] = np.ones((640,1), dtype=int)
            # b_img[:,0] = np.ones((640,1), dtype=int)


            img_array = np.array(b_img, dtype = int)
            x_left = []
            x_right = []
            y = np.arange(90)
            for i in range(90):
                row = img_array[i,:]
                try:
                    min_index = np.min(np.where(row > 0))
                    max_index = np.max(np.where(row > 0))
                except:
                    min_index = 320
                    max_index = 320
                x_left.append(min_index)
                x_right.append(max_index)
           
            try:
                a_l, b_l, r_l, p_l, std_err_l = stats.linregress(x_left, y)
                a_r, b_r, r_r, p_r, std_err_r = stats.linregress(x_right, y)
                pre_l = (45-b_l)/a_l
                pre_r = (45-b_r)/a_r
                center = (pre_r - pre_l)/2 + pre_l
                cv2.circle(img_show,(int(center), 295), 5, (255,0,0),3)
                cv2.circle(img_show,(int(pre_r), 295), 5, (255,0,0),3)
                cv2.circle(img_show,(int(pre_l), 295), 5, (255,0,0),3)
                cv2.line(img_show, (int(pre_r), 295), (int(pre_l), 295),(0,0,255), 3)
            except:
                center = 320
         


            error = 320 - center
            angle = PID(error, 0.41 ,0.0, 0.05) #0.15,0.0,0.04
            speed = control_speed(angle)
            print("Current_Angle = ",current_angle )
            #print("Angle = ",angle )
            #print("Speed = ",speed )
            
            message = bytes(f"{-angle} {speed}", "utf-8")
            s.sendall(message)

            #cv2.imshow("IMG", img_show)
            #cv2.imshow("binary", b_img)
            print("==============INCEPTION==============")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # cv2.imwrite("test.png", image)
                break
    finally:
        print('closing socket')
        s.close()
