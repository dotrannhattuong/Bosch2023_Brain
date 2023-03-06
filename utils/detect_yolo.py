import cv2
import torch
import random
import time
import numpy as np
import tensorrt as trt
from PIL import Image
from pathlib import Path
from collections import OrderedDict,namedtuple

# RealSense
from my_realsense import RealSense

class YOLO_DETECT:
    def __init__(self, engine_path='./yolov7-tiny-nms.trt', imgsz=(418,418), device='cuda:0'):
        self.__device = torch.device(device)
        self.__imgsz = imgsz
        
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
        self.__names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
        'hair drier', 'toothbrush']
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
        print('******* WARMUP *******')
        # warmup for 10 times
        for _ in range(20):
            tmp = torch.randn(1,3,418,418).to(self.__device)
            self._binding_addrs['images'] = int(tmp.data_ptr())
            self.context.execute_v2(list(self._binding_addrs.values()))

    def __call__(self, img):
        ####### Preprocessing #######
        im = self.preproc(img)
        
        #######    Predict    #######
        start = time.perf_counter()
        self._binding_addrs['images'] = int(im.data_ptr())
        self.context.execute_v2(list(self._binding_addrs.values()))

        print(f'Cost {time.perf_counter()-start} s')
        print(1/(time.perf_counter() - start), 'FPS')

        #######    Results    #######
        nums = self.__bindings['num_dets'].data
        boxes = self.__bindings['det_boxes'].data
        scores = self.__bindings['det_scores'].data
        classes = self.__bindings['det_classes'].data

        boxes = boxes[0,:nums[0][0]]
        scores = scores[0,:nums[0][0]]
        classes = classes[0,:nums[0][0]]

        return boxes, scores, classes

    def visualize(self, img, boxes, scores, classes, depth_frame=None):
        for box,score,cl in zip(boxes,scores,classes):
            box = self.postprocess(box,self.__ratio,self.__dwdh).round().int()
            name = self.__names[cl]
            color = self.__colors[name]
            name += ' ' + str(round(float(score),3))
            x1, y1 = box[:2]
            x2, y2 = box[2:]

            if depth_frame:
                d1, d2 = int((x1+x2)/2), int((y1+y2)/2)
                zDepth = depth_frame.get_distance(int(d1),int(d2))  # by default realsense returns distance in meters
                tl = 3 #line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
                tf = max(tl - 1, 1)  # font thickness
                cv2.putText(img, str(round((zDepth* 100 ),2))+" cm", (x1 + 200, y1), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

            cv2.rectangle(img,(int(x1),int(y1)), (int(x2),int(y2)),color,2)
            cv2.putText(img, name, (int(box[0]), int(box[1]) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, thickness=2)

        return img
        
if __name__ == "__main__":
    
    detector = YOLO_DETECT(engine_path='./yolov7-tiny-nms.trt', imgsz=(448,448))

    ####### Image #######
    img = cv2.imread('./inference/images/horses.jpg')

    # Detect
    boxes, scores, classes = detector(img)

    # Visualize
    img_pred = detector.visualize(img, boxes, scores, classes)

    # Write images
    cv2.imwrite('test.png', img_pred)

    ####### RealSense #######
    realsense = RealSense()

    try:
        while True:
            # Get image
            color_image, depth_frame, depth_colormap = realsense()

            # Detect
            boxes, scores, classes = detector(color_image)

            # Visualize
            img_pred = detector.visualize(color_image, boxes, scores, classes, depth_frame)

            # Show images
            cv2.imshow('RealSense', img_pred)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Stop streaming
        realsense.pipeline.stop()