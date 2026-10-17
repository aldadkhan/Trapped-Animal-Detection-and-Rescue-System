import os

import cv2
import numpy as np
from ultralytics import YOLO

FALLBACK_MODELS = ("yolo26n.pt", "yolo11n.pt", "yolov8n.pt")

# Extra animal-like COCO-adjacent labels (used if a custom model provides them)
ANIMAL_CLASSES = {
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "deer", "goat", "pig", "monkey", "rabbit",
}


class AnimalDetector:
    def __init__(self, model_path="yolo26n.pt"):
        self.model_path, self.model = self._load(model_path)
        self.animal_classes = ANIMAL_CLASSES

    @staticmethod
    def _load(model_path):
        """Try the requested model, then fall back to models that ultralytics
        can auto-download. Old ultralytics versions cannot read yolo26n.pt,
        which was the reason images were never analyzed."""
        tried = []
        candidates = [model_path] + [m for m in FALLBACK_MODELS if m != model_path]
        here = os.path.dirname(os.path.abspath(__file__))
        for name in candidates:
            if not name:
                continue
            path = name if os.path.isabs(name) else (
                os.path.join(here, name) if os.path.exists(os.path.join(here, name)) else name
            )
            try:
                return name, YOLO(path)
            except Exception as error:  # noqa: BLE001
                tried.append(f"{name}: {error}")
        raise RuntimeError("Could not load any YOLO model.\n" + "\n".join(tried))

    def detect(self, image, confidence=0.25):
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Input image is empty.")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        image = np.ascontiguousarray(image, dtype=np.uint8)

        # Downscale huge phone photos -> much faster inference
        h, w = image.shape[:2]
        scale = 1.0
        if max(h, w) > 1280:
            scale = 1280 / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)))

        results = self.model.predict(source=image, conf=confidence, imgsz=640, verbose=False)
        annotated = image.copy()
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                label = self.model.names[int(box.cls[0].item())]
                if label not in self.animal_classes:
                    continue
                score = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"{label}: {score:.2f}", (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                detections.append({"label": label, "confidence": score, "bbox": [x1, y1, x2, y2]})
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return annotated, detections
