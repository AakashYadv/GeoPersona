from ultralytics import YOLO
import numpy as np

class Detector:
    def __init__(self, model_name="yolov8s.pt", conf=0.5):
        self.model = YOLO(model_name)
        self.conf = conf

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]

        bboxes = []
        classes = []
        scores = []

        for det in results.boxes:

            cls = int(det.cls.cpu().numpy())
            score = float(det.conf.cpu().numpy())

            # Only PERSON class (0)
            if cls != 0:
                continue

            # Remove low-confidence detections
            if score < self.conf:
                continue

            xyxy = det.xyxy.cpu().numpy().flatten().tolist()

            bboxes.append(xyxy)
            classes.append(cls)
            scores.append(score)

        return bboxes, classes, scores
