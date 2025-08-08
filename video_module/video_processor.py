import cv2
import threading
from inference_module.inference_engine import InferenceEngine
from video_module.failsafe_watchdog import FailSafeWatchdog
import time

class VideoProcessor:
    def __init__(self, video_path, model_paths):
        self.cap = cv2.VideoCapture(video_path)
        self.model_paths = model_paths
        self.models = [InferenceEngine(path) for path in model_paths]
        self.latest_frame = None
        self.lock = threading.Lock()
        self.stop_flag = False
        self.watchdog = FailSafeWatchdog(timeout_seconds=5)

    def frame_reader(self):
        while not self.stop_flag:
            ret, frame = self.cap.read()
            if not ret:
                self.stop_flag = True
                break
            with self.lock:
                self.latest_frame = frame

    def run_inference(self):
        log_file = open("inference_log.txt", "w")
        log_file.write("Frame, Model, Detected, InferenceTime(ms)\n")
        frame_count = 0

        while not self.stop_flag:
            with self.lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None

            if frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1

            for model in self.models:
                count, boxes, inf_time = model.infer(frame)
                log_file.write(f"{frame_count}, {model.model_version}, {count}, {inf_time:.2f}\n")

                for (x1, y1, x2, y2, conf) in boxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Async Inference with Frame Skipping', frame)
            self.watchdog.heartbeat()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_flag = True
                break

        log_file.close()

    def start(self):
        reader_thread = threading.Thread(target=self.frame_reader)
        watchdog_thread = threading.Thread(target=self.watchdog.monitor, daemon=True)
        reader_thread.start()
        watchdog_thread.start()
        self.run_inference()
        reader_thread.join()
        self.cap.release()
        cv2.destroyAllWindows()