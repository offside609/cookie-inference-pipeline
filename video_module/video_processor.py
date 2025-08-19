import cv2
import threading
from inference_module.inference_engine import InferenceEngine
from video_module.failsafe_watchdog import FailSafeWatchdog
import time
import math
import queue
import os

class VideoProcessor:
    def __init__(self, video_path, model_paths):
        self.video_path = video_path
        self.model_paths = model_paths
        self.cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        self.models = [InferenceEngine(path) for path in model_paths]
        self.latest_frame = None
        self.lock = threading.Lock()
        self.stop_flag = False
        self.watchdog = FailSafeWatchdog(timeout_seconds=5)
        self.frame_queue = queue.Queue(maxsize=1)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.downscale = 0.5  # NEW: 1.0 keeps original; try 0.75 → 0.5

    def frame_reader(self):
        consecutive_fail = 0
        while not self.stop_flag:
            ret, frame = self.cap.read()
            if not ret:
                consecutive_fail += 1
                if consecutive_fail >= 30:
                    self.stop_flag = True
                    try:
                        self.frame_queue.put_nowait(None)
                    except Exception:
                        pass
                    break
                time.sleep(0.005)
                continue
            consecutive_fail = 0

            if self.frame_queue.full():
                try:
                    _ = self.frame_queue.get_nowait()
                except Exception:
                    pass
            try:
                self.frame_queue.put_nowait(frame)
            except Exception:
                pass

    def run_inference(self):
        log_file = open("inference_log.txt", "w")
        log_file.write("Frame, Model, Detected, InferenceTime(ms)\n")
        frame_count = 0
        writer = None
        out_path = "output_labeled.mp4"

        try:
            while not self.stop_flag:
                try:
                    frame = self.frame_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if frame is None:
                    break

                # NEW: resize input to reduce compute
                if hasattr(self, "downscale") and self.downscale and self.downscale != 1.0:
                    h0, w0 = frame.shape[:2]
                    new_w = max(1, int(w0 * self.downscale))
                    new_h = max(1, int(h0 * self.downscale))
                    print(new_w, new_h)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                if writer is None:
                    h, w = frame.shape[:2]
                    print(w, h)
                    raw_fps = self.cap.get(cv2.CAP_PROP_FPS)
                    fps = raw_fps
                    if not fps or math.isnan(fps) or fps < 1:
                        fps = 30.0
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
                    if not writer.isOpened():
                        fourcc = cv2.VideoWriter_fourcc(*"avc1")
                        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
                    if not writer.isOpened():
                        fourcc = cv2.VideoWriter_fourcc(*"XVID")
                        out_path = "output_labeled.avi"
                        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError(f"VideoWriter failed to open (fps={fps}, size={(w,h)})")

                frame_count += 1

                for model in self.models:
                    try:
                        count, boxes, inf_time = model.infer(frame)
                        log_file.write(f"{frame_count}, {model.model_version}, {count}, {inf_time:.2f}\n")
                        for (x1, y1, x2, y2, conf) in boxes:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    except Exception as e:
                        print(f"Error during inference with model {model.model_version}: {e}")
                        log_file.write(f"{frame_count}, {model.model_version}, ERROR, 0.0\n")

                cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                writer.write(frame)
                cv2.imshow('Async Inference with Frame Skipping', frame)
                self.watchdog.heartbeat()

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop_flag = True
                    break

        except Exception as e:
            print(f"Error in run_inference: {e}")
        finally:
            log_file.close()
            if writer is not None:
                writer.release()
            print(f"Video writer released. Total frames processed: {frame_count}")
                
    def start(self):
        reader_thread = threading.Thread(target=self.frame_reader, daemon=True)
        watchdog_thread = threading.Thread(target=self.watchdog.monitor, daemon=True)

        reader_thread.start()
        #watchdog_thread.start()

        # Keep GUI on main thread
        self.run_inference()

        reader_thread.join()
        self.cap.release()
        cv2.destroyAllWindows()