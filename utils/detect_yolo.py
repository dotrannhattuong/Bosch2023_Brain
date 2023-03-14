import cv2
import torch
import random
import time
import numpy as np
import tensorrt as trt
from PIL import Image
from pathlib import Path
from collections import OrderedDict,namedtuple
import copy
# RealSense
# from my_realsense import RealSense

class YOLO_DETECT:
    def __init__(self, engine_path='./best-nms.trt', imgsz=(418,418), device='cuda:0'):
        self.__device = torch.device(device)
        self.__imgsz = imgsz
        
        self.GHSVLOW = np.array([45, 100, 100])
        self.GHSVHIGH = np.array([90, 255, 255])
        self.YHSVLOW = np.array([20, 100, 100])
        self.YHSVHIGH = np.array([40, 255, 255])
        self.RHSVLOW = np.array([160,100,100])
        self.RHSVHIGH = np.array([180,255,255])
        self.RHSVLOW_1 = np.array([0,70,50])
        self.RHSVHIGH_1 = np.array([10,255,255])


        # Infer TensorRT Engine
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(logger, namespace="")
        with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        self.__bindings = OrderedDict()
        for index in range(model.num_bindings):
            name = model.get_binding_name(index)
            dtype = trt.nptype(model.get_binding_dtype(index))
            shape = tuple(model.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(self.__device)
            self.__bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
        self._binding_addrs = OrderedDict((n, d.ptr) for n, d in self.__bindings.items())
        self.context = model.create_execution_context()

        # Visualize
        self.__names = ['crosswalk_sign', 'highway_entrance_sign', 'highway_exit_sign', \
        'no_entry_road_sign', 'one_way_sign', 'parking_sign', 'priority_sign', 'round_about_sign', \
        'stop_sign', 'traffic_light', 'none']

        self.__colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(self.__names)}

        # warmup
        self.warmup()

    def letterbox(self, im, color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(self.__imgsz, int):
            self.__imgsz = (self.__imgsz, self.__imgsz)

        # Scale ratio (new / old)
        r = min(self.__imgsz[0] / shape[0], self.__imgsz[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = self.__imgsz[1] - new_unpad[0], self.__imgsz[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

        return im, r, (dw, dh)

    def preproc(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = img.copy()
        image, self.__ratio, self.__dwdh = self.letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)

        im = image.astype(np.float32)
        im = torch.from_numpy(im).to(self.__device)
        im/=255

        return im

    @staticmethod
    def postprocess(boxes, r, dwdh):
        dwdh = torch.tensor(dwdh*2).to(boxes.device)
        boxes -= dwdh
        boxes /= r

        return boxes

    def warmup(self):
        print('***** WARMUP *****')
        # warmup for 10 times
        for _ in range(20):
            tmp = torch.randn(1,3,418,418).to(self.__device)
            self._binding_addrs['images'] = int(tmp.data_ptr())
            self.context.execute_v2(list(self._binding_addrs.values()))

    def __call__(self, bg_removed, depth_colormap, vis = False):
        ####### Preprocessing #######
        im = self.preproc(bg_removed)
        
        #######    Predict    #######
        start = time.perf_counter()
        self._binding_addrs['images'] = int(im.data_ptr())
        self.context.execute_v2(list(self._binding_addrs.values()))

        # print(f'Cost {time.perf_counter()-start} s')
        # print(1/(time.perf_counter() - start), 'FPS')

        #######    Results    #######
        nums = self.__bindings['num_dets'].data
        boxes = self.__bindings['det_boxes'].data
        scores = self.__bindings['det_scores'].data
        classes = self.__bindings['det_classes'].data

        boxes = boxes[0,:nums[0][0]]
        scores = scores[0,:nums[0][0]]
        classes = classes[0,:nums[0][0]]
        #######    Information    #######
        # print(len(classes))
        score = 0
        log = 'None'
        if len(classes) == 0:
            class_id = 10
            name = self.__names[class_id]
            c1 = 0
            c2 = 0
            s = 0
            log = 'None'
        else:
            score = scores.cpu().numpy()[0]
            class_id = classes.cpu().numpy()[0]
            # print(1+score)
            if 1+score > 0.3:
                box = self.postprocess(boxes[0], self.__ratio, self.__dwdh).round().int()
                x1, y1 = box[:2]
                x2, y2 = box[2:]
                c1, c2 = int((x1+x2)/2), int((y1+y2)/2) # center bbox

                s = abs(x2-x1)*abs(y2-y1)
                # print('--'*10, s)
                if c1 < bg_removed.shape[1]//2 or s < 2500:
                    class_id = 10
                    name = self.__names[class_id]
                    c1 = 0
                    c2 = 0
                    s = 0
                    log = 'center / s'
                else:
                    name = self.__names[class_id]
                    # traffic light
                    if class_id == 9:
                        color_trafficlight = self.traffic_light_det(bg_removed, box)
                        class_id = 11 + color_trafficlight # do vang xanh
                        if class_id == 11:
                            name = 'red'
                        elif class_id == 12:
                            name = 'yellow'
                        elif class_id == 13:
                            name = 'green'   
                    if vis:
                        # print(name)
                        cv2.rectangle(bg_removed,(int(x1),int(y1)), (int(x2),int(y2)),(255, 0, 255),2)
                        cv2.putText(bg_removed, name, (int(box[0]), int(box[1]) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), thickness=2)
                log = 'ok'
            else:
                class_id = 10
                name = self.__names[class_id]
                c1 = 0
                c2 = 0
                s = 0
                log = "None"

        return boxes, 1+score, classes, class_id, bg_removed, (c1, c2), name, s, log

    # def visualize(self, img, boxes, scores, classes, depth_frame=None):
    #     if len(classes) != 0:
    #         box = self.postprocess(boxes[0], self.__ratio, self.__dwdh).round().int()
    #         name = self.__names[classes[0]]
    #         color = self.__colors[name]
    #         name += ' ' + str(round(float(score),3))
    #         x1, y1 = box[:2]
    #         x2, y2 = box[2:]
            
    #         if depth_frame:
    #             d1, d2 = int((x1+x2)/2), int((y1+y2)/2)
    #             zDepth = depth_frame.get_distance(int(d1),int(d2))  # by default realsense returns distance in meters
    #             tl = 3 #line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    #             tf = max(tl - 1, 1)  # font thickness
    #             cv2.putText(img, str(round((zDepth* 100 ),2))+" cm", (x1 + 200, y1), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    #         cv2.rectangle(img,(int(x1),int(y1)), (int(x2),int(y2)),color,2)
    #         cv2.putText(img, name, (int(box[0]), int(box[1]) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, thickness=2)
        
    #     return img

    # def visualize(self, img, boxes, scores, classes, depth_frame=None):
        # for box,score,cl in zip(boxes,scores,classes):
        #     box = self.postprocess(box,self.__ratio,self.__dwdh).round().int()
        #     name = self.__names[cl]
        #     color = self.__colors[name]
        #     name += ' ' + str(round(float(score),3))
        #     x1, y1 = box[:2]
        #     x2, y2 = box[2:]
            
        #     if depth_frame:
        #         d1, d2 = int((x1+x2)/2), int((y1+y2)/2)
        #         zDepth = depth_frame.get_distance(int(d1),int(d2))  # by default realsense returns distance in meters
        #         tl = 3 #line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        #         tf = max(tl - 1, 1)  # font thickness
        #         cv2.putText(img, str(round((zDepth* 100 ),2))+" cm", (x1 + 200, y1), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

        #     cv2.rectangle(img,(int(x1),int(y1)), (int(x2),int(y2)),color,2)
        #     cv2.putText(img, name, (int(box[0]), int(box[1]) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, thickness=2)
        # return img

    def traffic_light_det(self, img, box): # boxes: x,y,w,h
        '''
        input: img, box
        output: 0: red, 1: yellow, 2: green
        '''
        # TODO: Check boxes
        x1, y1 = box[:2]
        x2, y2 = box[2:]
        img_crop = img[int(y1):int(y2), int(x1):int(x2)]
        img_hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
        maskg = cv2.inRange(img_hsv, self.GHSVLOW, self.GHSVHIGH)
        masky = cv2.inRange(img_hsv, self.YHSVLOW, self.YHSVHIGH)
        maskr_1 = cv2.inRange(img_hsv, self.RHSVLOW, self.RHSVHIGH)
        maskr_2 = cv2.inRange(img_hsv, self.RHSVLOW_1, self.RHSVHIGH_1)
        maskr = maskr_1 | maskr_2

        area = [self.check(mask) for mask in [maskr, masky, maskg]]
        index = area.index(max(area))
        return index

    # def traffic_light_det(self, img, boxes): # boxes: x,y,w,h
    #     '''
    #     input: img, box
    #     output: 0: red, 1: yellow, 2: green
    #     '''
    #     # TODO: Check boxes

    #     # box = self.postprocess(boxes,self.__ratio,self.__dwdh).round().int()

    #     dwdh = torch.tensor(self.__dwdh*2).to(boxes.device)
    #     _boxes = copy.deepcopy(boxes)
    #     _boxes -= dwdh
    #     _boxes /= self.__ratio
    #     _boxes = _boxes.round().int()
        
    #     x1, y1 = _boxes[:2]
    #     x2, y2 = _boxes[2:]

    #     img_crop = img[int(y1):int(y2), int(x1):int(x2)]
    #     img_hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
    #     cv2.imwrite("img_crop.png", img_crop)
    #     maskg = cv2.inRange(img_hsv, self.GHSVLOW, self.GHSVHIGH)
    #     masky = cv2.inRange(img_hsv, self.YHSVLOW, self.YHSVHIGH)
    #     maskr_1 = cv2.inRange(img_hsv, self.RHSVLOW, self.RHSVHIGH)
    #     maskr_2 = cv2.inRange(img_hsv, self.RHSVLOW_1, self.RHSVHIGH_1)
    #     maskr = maskr_1 | maskr_2

    #     area = [self.check(mask) for mask in [maskr, masky, maskg]]
    #     index = area.index(max(area))
    #     return index

    @staticmethod
    def check(mask):
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            return 0
        max_contour = max(contours, key = cv2.contourArea)
        area = cv2.contourArea(max_contour)
        return area

    

if __name__ == "__main__":
    
    detector = YOLO_DETECT(engine_path='./weights/yolo/best-v1-nms.trt', imgsz=(448,448))

    ####### Image #######
    img = cv2.imread('2134.png')

    # Detect
    boxes, scores, classes = detector(img)

    print(boxes, scores, classes, img.shape)
    
    cv2.imwrite('b.png', img)
    # Demo traffic light
    if 9 in list(classes.cpu().numpy()):
        box_trafficlight = boxes[list(classes.cpu().numpy()).index(9)]
        color_trafficlight = detector.traffic_light_det(img, box_trafficlight)
        print(color_trafficlight)

    # Visualize
    cv2.imwrite('a.png', img)
    print(boxes, scores, classes, img.shape)
    img_pred = detector.visualize(img, boxes, scores, classes)

    
    # Write images
    cv2.imwrite('test.png', img_pred)
    

    ####### RealSense #######
    # realsense = RealSense()

    # while True:
    #     # Get image
    #     color_image, depth_frame, depth_colormap = realsense()

    #     # Detect
    #     boxes, scores, classes = detector(color_image)

    #     # Visualize
    #     img_pred = detector.visualize(color_image, boxes, scores, classes, depth_frame)
    #     print(boxes)
    #     cv2.imwrite('pred.png', img_pred)
        # Show images
        # cv2.imshow('RealSense', img_pred)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     # Stop streaming
        #     realsense.pipeline.stop()
        #     break