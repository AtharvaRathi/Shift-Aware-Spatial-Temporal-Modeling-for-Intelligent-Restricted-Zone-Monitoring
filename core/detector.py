import torch
from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_path='yolov8x.pt'):
        """
        yolov8x = Extra-Large model.
        Requires ~12 GB VRAM — perfect for the A5000 (24 GB).
        Falls back to yolov8n on low-VRAM machines.
        """
        vram_gb = 0
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb < 8:
            print(f'[Detector] Low VRAM ({vram_gb:.1f} GB), falling back to yolov8n.pt')
            model_path = 'yolov8n.pt'
        else:
            print(f'[Detector] {vram_gb:.1f} GB VRAM detected — using {model_path}')
        self.model = YOLO(model_path)
        self.half  = torch.cuda.is_available()  # FP16 on GPU, FP32 on CPU

    def detect(self, frame):
        # class 0 = person, 24 = backpack, 26 = handbag
        results = self.model.predict(frame, classes=[0, 24, 26],
                                     half=self.half, verbose=False)
        return results[0]

    def track(self, frame, persist=True):
        results = self.model.track(frame, classes=[0, 24, 26], persist=persist,
                                   half=self.half, verbose=False)
        return results[0]
