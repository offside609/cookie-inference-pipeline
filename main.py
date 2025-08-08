from video_module.video_processor import VideoProcessor

if __name__ == "__main__":
    model_paths = [
        "models/yolo11n.pt",               # Pretrained  # Fine-tuned (replace after training)
        "models/yolo11n_int8.tflite"   # Quantized (after quantization)
    ]
    video_processor = VideoProcessor("Cookie_on_belt.mp4", model_paths)
    video_processor.start()



