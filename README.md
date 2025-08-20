## FastInfer

A minimal, fast video inference pipeline that reads frames asynchronously, runs object detection, overlays results, and writes annotated video.I have taken cookies running on a belt as example. Results are still not complete. Inference pipeline takes 300 ms to run with int8 tflite model with 640 by 640 input. Need to come to 40 ms to process video online and completely. That's why currently it is outputting final frame only as I am dropping all the old frames in buffer. 

Aim : To show understanding of problem

Future work in 7 days : 
- Smalled tflite model with 320 input size
- Better frame skipping logic.
- finetune tflite model for better prediction.
- Add multiple workers.

### Features
- Queue-based capture → inference pipeline (drops backlog; processes newest frame).
- Supports Ultralytics YOLO `.pt` and TFLite `.tflite` models.
- Writes annotated output video and logs per-frame inference results.
- macOS-friendly UI (OpenCV window on main thread).

### Repository layout
- `main.py`: entry point to run the pipeline
- `video_module/video_processor.py`: threaded capture and inference loop
- `inference_module/inference_engine.py`: backend abstraction for YOLO/TFLite
- `models/`: place your models here (ignored by git)

### Requirements
- Python 3.10+
- macOS or Linux
- Packages (install below): OpenCV, ultralytics, tensorflow (for TFLite), numpy

### Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you don’t have a `requirements.txt` synced, install directly:
```bash
pip install opencv-python ultralytics tensorflow numpy
```

### Usage
1) Put your model in `models/`:
   - YOLO PyTorch: `models/yolo11n.pt`
   - TFLite INT8: `models/yolo11n_int8.tflite`
  
   - I have already provided a copy. Though it is not tuned on custom data.

2) Update `main.py` if needed (choose model and input video):
```python
model_paths = [
    "models/yolo11n.pt",
    "models/yolo11n_int8.tflite"
]
video_processor = VideoProcessor("Cookie_on_belt.mp4", model_paths)
video_processor.start()
```

3) Run:
```bash
python main.py
```
- Window shows live annotated frames
- Output video is written as `output_labeled.mp4`
- Per-frame logs in `inference_log.txt`

### Performance controls
- Resize before inference: in `video_module/video_processor.py`, adjust
```python
self.downscale = 0.5  # e.g., 0.75, 0.5, 0.33
```
- YOLO `.pt` path: internally capped to a smaller `imgsz` to avoid upscaling.
- TFLite path: frame is resized to the model’s input tensor size. To avoid upscaling, export a smaller TFLite model (see below).



### Notes
- Live sources: capture can briefly fail; the reader tolerates short glitches and only stops after repeated failures.
- Writer: uses fallback codecs (`mp4v`, `avc1`, `XVID`) if needed. Try VLC if default player fails to open.
- UI: on macOS keep `imshow`/`waitKey` on the main thread (already handled by `VideoProcessor.start`).

### Troubleshooting
- Only last frame seems processed: this is expected with a fast source and slower model. The queue keeps latency low by processing the newest frame. To process every frame, change logic to use an unbounded queue (accept higher latency) or speed up the model (smaller input/imgsz, INT8, GPU).
- TFLite shows shape 640×640 and is slow: export the model at a smaller `imgsz` (e.g., 384 or 320) and use that `.tflite`.
- Push to GitHub: artifacts are ignored via `.gitignore` (`models/`, logs, videos, caches).

### License
MIT (or your chosen license)# cookie-inference-pipeline
