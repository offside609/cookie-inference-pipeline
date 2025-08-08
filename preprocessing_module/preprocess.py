import cv2
import numpy as np

class Preprocessor:
    def __init__(self, target_size=(224, 224), normalize=True):
        self.target_size = target_size
        self.normalize = normalize

    def preprocess_frame(self, frame):
        # Resize frame (bilinear interpolation)
        frame_resized = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Normalize pixel values to [0, 1]
        if self.normalize:
            frame_normalized = frame_rgb.astype(np.float32) / 255.0
        else:
            frame_normalized = frame_rgb.astype(np.float32)

        return frame_normalized
