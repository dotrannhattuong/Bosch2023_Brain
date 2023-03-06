# Pytorch to TensorRT with NMS (and inference)
```
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
python3 export.py --weights ./yolov7-tiny.pt --device 0 --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640
git clone https://github.com/Linaom1214/tensorrt-python.git
python3 ./tensorrt-python/export.py -o yolov7-tiny.onnx -e yolov7-tiny-nms.trt -p fp16
```

# [Inference](https://github.com/dotrannhattuong/Bosch2023_Brain/blob/main/utils/detect_yolo.py)
```
from detect_yolo import YOLO_DETECT

detector = YOLO_DETECT(engine_path='./yolov7-tiny-nms.trt', imgsz=(640,640)) # Chú ý image size

img = cv2.imread('./inference/images/horses.jpg')
boxes, scores, classes = detector(img)

img_pred = detector.visualize(img, boxes, scores, classes)

cv2.imwrite('test.png', img_pred)
```

# Reference
[yolov7](https://github.com/WongKinYiu/yolov7)