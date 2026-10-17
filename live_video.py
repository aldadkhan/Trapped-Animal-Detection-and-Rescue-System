from threading import Lock

import av
import cv2
import numpy as np
from streamlit_webrtc import VideoProcessorBase

from detector import AnimalDetector


class LiveAnimalVideoProcessor(VideoProcessorBase):
    def __init__(self, model_path, confidence):
        self.detector = AnimalDetector(model_path)
        self.confidence = confidence
        self._frame_lock = Lock()
        self._latest_frame = None
        self._count = 0
        self._last = (None, [])

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        with self._frame_lock:
            self._latest_frame = image.copy()
        # Run detection every 3rd frame for speed, reuse boxes between
        self._count += 1
        if self._count % 3 == 1 or self._last[0] is None:
            annotated, detections = self.detector.detect(image, confidence=self.confidence)
            self._last = (annotated, detections)
        annotated, detections = self._last
        if annotated.shape != image.shape:
            annotated = cv2.resize(annotated, (image.shape[1], image.shape[0]))
        annotated = annotated.copy()
        label = (f"{len(detections)} animal(s): " + ", ".join(d["label"] for d in detections[:3])
                 if detections else "No supported animal detected")
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(annotated, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    def get_latest_frame(self):
        with self._frame_lock:
            return None if self._latest_frame is None else np.copy(self._latest_frame)
