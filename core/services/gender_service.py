"""Gender detection service for person crops.

Uses OpenCV's YuNet face detector (ONNX) and a pre-trained GoogLeNet gender
classifier (ONNX via ONNX Runtime).  Models are downloaded automatically on
first use and cached under MEDIA_ROOT/models.  If the models cannot be loaded,
no face is detected, or the classifier is not confident enough, the service
returns 'unknown' so callers can fall back to pose-based heuristics.
"""

import logging
import os
import threading
import urllib.error
import urllib.request

import cv2
import numpy as np
from django.conf import settings

logger = logging.getLogger(__name__)

# Pre-trained model URLs.
FACE_DETECTOR_URL = (
    'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/'
    'face_detection_yunet_2023mar.onnx'
)
GENDER_MODEL_URL = (
    'https://huggingface.co/onnxmodelzoo/gender_googlenet/resolve/main/'
    'gender_googlenet.onnx'
)


def _model_dir():
    return os.path.join(str(settings.MEDIA_ROOT), 'models')


def _download_file(url, dest_path, timeout=180):
    """Download a file atomically if it does not already exist. Returns True on success."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return True
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    temp_path = f'{dest_path}.tmp'
    try:
        logger.info('Downloading %s ...', url)
        # Use urlopen with a timeout so a hanging download cannot block forever.
        with urllib.request.urlopen(url, timeout=timeout) as response:
            with open(temp_path, 'wb') as handle:
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
        # Atomic move so other threads never see a partial file.
        os.replace(temp_path, dest_path)
        return os.path.exists(dest_path) and os.path.getsize(dest_path) > 0
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        logger.warning('Failed to download %s: %s', url, exc)
        for path in (temp_path, dest_path):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        return False


class GenderService:
    """Thread-safe singleton that detects gender from face crops."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Avoid re-initializing if __new__ returned an existing instance.
        if getattr(self, '_init_done', False):
            return
        self._init_done = True

        self._inference_lock = threading.Lock()
        self._init_lock = threading.Lock()
        self._face_detector = None
        self._gender_session = None
        self._ready = False
        self._load_error = ''
        self._initialized = False

    def _initialize(self):
        """Lazy-load and cache the face detector and gender classifier."""
        with self._init_lock:
            if self._initialized:
                return
            self._initialized = True

            face_model = os.path.join(_model_dir(), 'face_detection_yunet_2023mar.onnx')
            gender_model = os.path.join(_model_dir(), 'gender_googlenet.onnx')

            if not _download_file(FACE_DETECTOR_URL, face_model):
                self._load_error = 'face detector model unavailable'
                return
            if not _download_file(GENDER_MODEL_URL, gender_model):
                self._load_error = 'gender classifier model unavailable'
                return

            try:
                # YuNet face detector from OpenCV zoo.  Input size is updated
                # per-image via setInputSize.
                self._face_detector = cv2.FaceDetectorYN_create(face_model, '', (320, 320))

                # ONNX Runtime session for the gender classifier.
                import onnxruntime as ort

                self._gender_session = ort.InferenceSession(
                    gender_model, providers=['CPUExecutionProvider']
                )
                self._ready = True
                logger.info('GenderService loaded successfully')
            except Exception as exc:
                self._load_error = str(exc)
                logger.warning('Failed to load gender detection models: %s', exc)

    @property
    def ready(self):
        """Trigger model download/load and return whether inference is ready."""
        self._initialize()
        return self._ready

    @property
    def load_error(self):
        self._initialize()
        return self._load_error

    def detect_face(self, image, confidence_threshold=0.6):
        """Return the largest detected face crop from `image`, or None."""
        self._initialize()
        if not self._ready or self._face_detector is None:
            return None
        if image is None or image.size == 0:
            return None

        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return None

        with self._inference_lock:
            self._face_detector.setInputSize((w, h))
            retval, faces = self._face_detector.detect(image)

        if faces is None or len(faces) == 0:
            return None

        best_face = None
        best_area = 0
        for face in faces:
            score = float(face[-1])
            if score < confidence_threshold:
                continue
            x, y, fw, fh = map(int, face[:4])
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + fw), min(h, y + fh)
            if x2 <= x1 or y2 <= y1:
                continue
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_face = image[y1:y2, x1:x2]

        return best_face

    def classify_gender(self, face_crop, min_confidence=0.65):
        """Classify gender from a face crop.

        Returns one of: 'male', 'female', 'unknown'.
        """
        self._initialize()
        if not self._ready or self._gender_session is None:
            return 'unknown'
        if face_crop is None or face_crop.size == 0:
            return 'unknown'

        try:
            import onnxruntime as ort

            # GoogLeNet gender net expects 224x224 BGR with mean subtraction.
            blob = cv2.dnn.blobFromImage(
                face_crop,
                scalefactor=1.0,
                size=(224, 224),
                mean=(104, 117, 123),
                swapRB=False,
                crop=False,
            )

            input_name = self._gender_session.get_inputs()[0].name
            outputs = self._gender_session.run(None, {input_name: blob})
            logits = outputs[0][0]

            # Softmax to get probabilities.  Index 0 = male, index 1 = female.
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            male_prob = float(probs[0])
            female_prob = float(probs[1])
            if max(male_prob, female_prob) < min_confidence:
                return 'unknown'
            return 'male' if male_prob > female_prob else 'female'
        except Exception as exc:
            logger.warning('Gender classification failed: %s', exc)
            return 'unknown'

    def detect_gender(self, frame, x1=None, y1=None, x2=None, y2=None, min_confidence=0.65):
        """Detect gender from a full frame or a cropped person region.

        If `x1`, `y1`, `x2`, `y2` are provided, only that region is used.
        Returns one of: 'male', 'female', 'unknown'.
        """
        try:
            if x1 is not None:
                h, w = frame.shape[:2]
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                if x2 <= x1 or y2 <= y1:
                    return 'unknown'
                crop = frame[y1:y2, x1:x2]
            else:
                crop = frame

            face = self.detect_face(crop)
            if face is None or face.size == 0:
                return 'unknown'
            return self.classify_gender(face, min_confidence=min_confidence)
        except Exception as exc:
            logger.warning('detect_gender failed: %s', exc)
            return 'unknown'


_gender_service = None
_gender_service_lock = threading.Lock()


def get_gender_service():
    """Return the singleton GenderService instance."""
    global _gender_service
    if _gender_service is None:
        with _gender_service_lock:
            if _gender_service is None:
                _gender_service = GenderService()
    return _gender_service
