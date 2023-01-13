import time
import numpy as np
from scipy import stats
import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import ward, fcluster
from scipy.spatial.distance import pdist

from collections import Counter

class Lane_Detection:
    def __init__(self):
        self.__width = 220

    def __call__(self, image):
        image = self.image_processing(image)

        ####### Edges Detection #######
        edges = self.edges_detection(image)

        ####### Lane Processing #######
        center = self.lane_processing(edges)

        return center, self.__img_show
    
    def image_processing(self, image):
        image = cv2.resize(image, (640,360))
        self.__img_show = image.copy()
        image = image[250:,:,:]

        return image

    def edges_detection(self, image):
        img = cv2.GaussianBlur(image, (7,7), 0)
        # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, mask = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
        edges = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        # edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=10)
        edges = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)

        return edges

    def lane_processing(self, edges):
        img_array = np.array(edges, dtype = int)
        x_left = []
        y_left = []
        x_right = []
        y_right = []

        for i in range(edges.shape[0]):
            row = img_array[i,:]
            try:
                ####### Tìm lane trái - lane phải #######
                index = np.where(row > 0)[0]
                min_index = np.min(index)
                max_index = np.max(index)

                ####### Điều kiện với đường 2 làn (self.__width là độ rộng đường của 1 làn) #######
                if max_index - min_index >= self.__width:
                    ####### CLUSTERING #######
                    class_id = self.cluster(index) # class từ 0->n
                    
                    ####### Vạch đi bộ có 9 vạch -> đổi lại lane trái bằng lane chính giữa của vạch đi bộ là line thứ 5 #######
                    if max(class_id) == 9: 
                        new_min_index = index[class_id==5].ravel()[0]

                        if max_index - new_min_index <= self.__width:
                            min_index = new_min_index

                        else:
                            continue
                    
                    ####### Vạch đi bộ có noise line của biển báo => xử lý lane trái như TH trên còn lane phải trừ đi 2 class id #######
                    elif max(class_id) > 9:
                        end_cluster = index[class_id==max(class_id)].ravel()
                        if end_cluster[-1] >= 600:
                            new_max_index = index[class_id==max(class_id)-1].ravel()[-1]
                            new_min_index = index[class_id==5].ravel()[0]
                            
                            if new_max_index - new_min_index <= self.__width:
                                max_index = new_max_index
                                min_index = new_min_index

                            else:
                                continue
                        else: 
                            continue
                    
                    ####### Đường có 3 lane trở lên => lane trái = lane phải - 1 lane #######
                    elif max(class_id) > 2:
                        if max_index <= 600:
                            new_min_index = index[class_id==max(class_id)-1].ravel()[0]

                            if 100 <= max_index - new_min_index <= self.__width:
                                min_index = new_min_index
                            else: 
                                continue
                        else: 
                            continue

                    ####### Đường có 2 lane #######
                    else:
                        y_right.append(i)
                        x_right.append(max_index)
                        continue

                if max_index >= 300:
                    y_right.append(i)
                    x_right.append(max_index)

                if min_index <= 300:
                    y_left.append(i)
                    x_left.append(min_index)

            except:
                pass
        
        ######## Linear Regression ########
        try:
            a_r, b_r, r_r, p_r, std_err_r = stats.linregress(x_right, y_right)
            pre_r = (45-b_r)/a_r

            if len(x_left) != 0:
                a_l, b_l, r_l, p_l, std_err_l = stats.linregress(x_left, y_left)
                pre_l = (45-b_l)/a_l
            else:
                pre_l = pre_r - (self.__width - 60)

            ######## TH đường 1 làn nhưng rộng ########
            if pre_r - pre_l > self.__width + 50 or pre_r - pre_l <= 0:
                pre_l = pre_r - (self.__width - 60)

            center = (pre_r + pre_l)/2

            ######## Visualize ########
            cv2.circle(self.__img_show,(int(center), 295), 5, (255,0,0),3)
            cv2.circle(self.__img_show,(int(pre_r), 295), 5, (255,0,0),3)
            cv2.circle(self.__img_show,(int(pre_l), 295), 5, (255,0,0),3)
            cv2.line(self.__img_show, (int(pre_r), 295), (int(pre_l), 295),(0,0,255), 3)
            
        except:
            center = 320
        
        return center

    def cluster(self, index):
        X = index.reshape(-1, 1)

        Z = ward(pdist(X))
        y = fcluster(Z, t=50, criterion='distance')
        
        class_id = np.copy(y)
        counts = Counter(y).values()
        start = 0
        for i, c in enumerate(counts):
            class_id[start:start+c] = i+1
            start += c
        
        return class_id

if __name__ == "__main__":
    lane_det = Lane_Detection()

    size = (640, 360)

    # result = cv2.VideoWriter('lane.avi', 
    #                         cv2.VideoWriter_fourcc(*'MJPG'),
    #                         40, size)

    cap = cv2.VideoCapture("./record_3.avi")

    while(cap.isOpened()):
        ret, frame = cap.read()
        h, w, c = frame.shape
        frame = frame[400:600, w//2:, :]

        if ret == True:
            #====================segment==============================
            center, img_show = lane_det(frame)

            # result.write(img_show)

            cv2.imshow("img_show", img_show)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else: 
            break

    cap.release()
    # result.release()

    cv2.destroyAllWindows()
 