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
        h, w = frame.shape[:2]
        # Limit YOLO's internal resize so it doesn't upscale back toward 640
        target = min(max(h, w), 480)  # try 480; then 384/320 for more speed

        start_time = time.time()
        results = self.model(frame, imgsz=target, conf=self.confidence_threshold, verbose=False)
        inference_time = (time.time() - start_time) * 1000  # ms
        speed = results[0].speed  # dict with 'preprocess', 'inference', 'postprocess'
        print(f"Speed: {speed['preprocess']:.1f}ms preprocess, {speed['inference']:.1f}ms inference, {speed['postprocess']:.1f}ms postprocess per image at shape {results[0].orig_shape}")

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
        # Measure preprocess
        t0 = time.time()
        input_info = self.input_details[0]
        input_shape = input_info['shape']
        input_dtype = input_info['dtype']

        # Determine expected layout and resize accordingly
        if len(input_shape) == 4 and input_shape[-1] == 3:  # NHWC
            target_h, target_w = int(input_shape[1]), int(input_shape[2])
            is_nhwc = True
        elif len(input_shape) == 4 and input_shape[1] == 3:  # NCHW
            target_h, target_w = int(input_shape[2]), int(input_shape[3])
            is_nhwc = False
        else:
            # Fallback assume NHWC
            target_h, target_w = int(input_shape[1]), int(input_shape[2])
            is_nhwc = True
        print(target_h, target_w)

        # Resize and convert to RGB
        resized = cv2.resize(frame, (target_w, target_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Prepare batch dimension and channel order
        if is_nhwc:
            batched = np.expand_dims(rgb, axis=0)
        else:
            batched = np.expand_dims(np.transpose(rgb, (2, 0, 1)), axis=0)

        # Convert dtype appropriately
        if input_dtype == np.float32:
            input_data = batched.astype(np.float32) / 255.0
        elif input_dtype == np.uint8:
            input_data = batched.astype(np.uint8)
        elif input_dtype == np.int8:
            scale, zero_point = input_info.get('quantization', (1.0, 0)) or (1.0, 0)
            normalized = batched.astype(np.float32) / 255.0
            quantized = np.round(normalized / (scale if scale != 0 else 1.0) + zero_point)
            input_data = np.clip(quantized, -128, 127).astype(np.int8)
        else:
            input_data = batched.astype(np.float32) / 255.0

        t1 = time.time()  # end preprocess

        # Inference
        self.interpreter.set_tensor(input_info['index'], input_data)
        t2 = time.time()
        self.interpreter.invoke()
        t3 = time.time()
        inference_time = (t3 - t2) * 1000.0  # ms

        # Postprocess
        count = 0
        boxes = []
        try:
            outputs = []
            for output_detail in self.output_details:
                output_data = self.interpreter.get_tensor(output_detail['index'])
                outputs.append(output_data)

            if len(outputs) >= 2:
                boxes_output = outputs[0]  # [1, num_detections, 4]
                scores_output = outputs[1]  # [1, num_detections]
                orig_h, orig_w = frame.shape[:2]

                for i in range(scores_output.shape[1]):
                    confidence = float(scores_output[0, i])
                    if confidence >= self.confidence_threshold:
                        box = boxes_output[0, i]  # [y1, x1, y2, x2]
                        y1, x1, y2, x2 = box
                        x1 = int(x1 * orig_w)
                        y1 = int(y1 * orig_h)
                        x2 = int(x2 * orig_w)
                        y2 = int(y2 * orig_h)
                        x1 = max(0, min(x1, orig_w))
                        y1 = max(0, min(y1, orig_h))
                        x2 = max(0, min(x2, orig_w))
                        y2 = max(0, min(y2, orig_h))
                        boxes.append((x1, y1, x2, y2, confidence))
                        count += 1
        except Exception as e:
            print(f"Error processing TFLite outputs: {e}")
            pass

        t4 = time.time()  # end postprocess

        preprocess_ms = (t1 - t0) * 1000.0
        postprocess_ms = (t4 - t3) * 1000.0
        print(f"Speed: {preprocess_ms:.1f}ms preprocess, {inference_time:.1f}ms inference, {postprocess_ms:.1f}ms postprocess per image at shape ({target_h}, {target_w})")

        return count, boxes, inference_time