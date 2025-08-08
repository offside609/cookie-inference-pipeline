from ultralytics import YOLO
import tensorflow as tf
import cv2
import numpy as np
import os
import time

class InferenceEngine:
    def __init__(self, model_path, confidence_threshold=0.4):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.backend = None
        self.model_version = os.path.basename(model_path)

        if model_path.endswith('.pt'):
            self.backend = 'yolo'
            self.model = YOLO(model_path)
        elif model_path.endswith('.tflite'):
            self.backend = 'tflite'
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        else:
            raise ValueError("Unsupported model format.")

    def infer(self, frame):
        if self.backend == 'yolo':
            return self._infer_yolo(frame)
        elif self.backend == 'tflite':
            return self._infer_tflite(frame)

    def _infer_yolo(self, frame):
        start_time = time.time()
        results = self.model(frame)
        inference_time = (time.time() - start_time) * 1000  # ms

        detections = results[0].boxes
        count = 0
        boxes = []

        for box in detections:
            conf = float(box.conf[0])
            if conf >= self.confidence_threshold:
                count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2, conf))

        return count, boxes, inference_time

    def _infer_tflite(self, frame):
        start_time = time.time()
        input_data = cv2.resize(frame, (self.input_details[0]['shape'][2], self.input_details[0]['shape'][1]))
        input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(input_data, axis=0).astype(np.uint8)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        inference_time = (time.time() - start_time) * 1000  # ms

        count = 0
        boxes = []

        return count, boxes, inference_time
