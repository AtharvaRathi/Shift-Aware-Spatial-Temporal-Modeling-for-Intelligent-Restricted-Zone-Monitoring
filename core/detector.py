import torch
from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_path='yolov8x.pt'):
        """
        Modified to support Apple Silicon (MPS) for massive speedups!
        """
        self.device = 'cpu'
        
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if vram_gb < 8:
                print(f'[Detector] Low CUDA VRAM ({vram_gb:.1f} GB), falling back to yolov8n.pt')
                model_path = 'yolov8n.pt'
            else:
                print(f'[Detector] {vram_gb:.1f} GB VRAM detected — using {model_path}')
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            print('[Detector] Apple Silicon (MPS GPU) detected! Forcing hardware acceleration and yolov8n.pt for speed.')
            model_path = 'yolov8n.pt'
            self.device = 'mps'
        else:
            print('[Detector] No GPU detected. Falling back to CPU and yolov8n.pt')
            model_path = 'yolov8n.pt'
            
        self.model = YOLO(model_path)

    def detect(self, frame):
        # class 0 = person, 24 = backpack, 26 = handbag
        results = self.model.predict(frame, classes=[0, 24, 26], verbose=False, device=self.device, imgsz=640)
        return results[0]

    def track(self, frame, persist=True):
        results = self.model.track(frame, classes=[0, 24, 26], persist=persist, verbose=False, device=self.device, imgsz=640)
        return results[0]
