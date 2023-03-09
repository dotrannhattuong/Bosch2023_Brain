import cv2
import numpy as np

def lane_processing(color_image):
    color_image = color_image[200:,:,:]
    cv2.imshow('color_image', color_image)
    mask = np.zeros(color_image.shape, np.uint8)
    blurred = cv2.medianBlur(color_image, 3)  
    image_gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    image_gray = 255-image_gray 
    
    bw1 = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 61, 20)
    _, bw2 = cv2.threshold(image_gray, 90, 255, cv2.THRESH_BINARY_INV)
    bw = bw1*bw2
    dst_bw2 = cv2.erode(bw, None, iterations=1)
    dst_bw2 = cv2.dilate(dst_bw2, None, iterations=1)
    contours, _ = cv2.findContours(image=dst_bw2, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

    for con in contours:
        area = cv2.contourArea(con)
        if area>100 :
            cv2.fillPoly(mask, pts =[con], color=(255,255,255))

    mask = cv2.resize(mask, (160, 80))
    return mask

if __name__ == "__main__":
    color_img = cv2.imread('./images/448.png')

    edge = lane_processing(color_img)
    print(edge.shape)
    cv2.imshow('results', edge)
    cv2.waitKey()
    cv2.destroyAllWindows()

