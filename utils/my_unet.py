import cv2
import onnxruntime
import numpy as np

class Unet:
    def __init__(self, weights='bosch_3c_sof.onnx'):
        # ---------- Seg ------------#
        self.session_lane = onnxruntime.InferenceSession(weights, None, providers=['CPUExecutionProvider'])
        self.input_name_lane = self.session_lane.get_inputs()[0].name

    def road_lines(self, color_image, session, inputname):
        image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        # Crop ảnh lại, lấy phần ảnh có làn đườngs
        image = image[160:, :, :]
        image = cv2.resize(image, (160, 120))
        image = image/255
        image = np.array(image, dtype=np.float32)
        # print(image.shape)
        # image = image.transpose(2, 0, 1)
        # print(image.shape)
        image = image[None, :, :, :]
        prediction = session.run(None, {inputname: image})
        prediction = np.squeeze(prediction)

        # prediction = np.argmax(prediction, -1)
        # prediction = np.where(prediction == 1, 255, prediction)
        # prediction = np.where(prediction == 2, 255, prediction)
        # prediction = prediction.astype(np.uint8)

        prediction = np.where(prediction > 0.5, 255, 0)
        prediction = prediction.astype(np.uint8)

        return prediction

    @staticmethod
    def remove_small_contours(image):
        image_binary = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        contours = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        # new_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3*len(contours)//4]
        mask = cv2.drawContours(image_binary, [max(contours, key=cv2.contourArea)], -1, (255, 255, 255), -1)
        # mask = cv2.drawContours(image_binary, new_contours, -1, (255, 255, 255), -1)
        image_remove = cv2.bitwise_and(image, image, mask=mask)
        return image_remove

    def __call__(self, color_image):
        img = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        pred = self.road_lines(img, session=self.session_lane,inputname=self.input_name_lane)

        # Remove Contours
        # pred = self.remove_small_contours(pred)

        return pred