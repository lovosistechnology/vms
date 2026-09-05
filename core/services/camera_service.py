import math
import os
import sys
import threading
import time
from django.conf import settings

# Silence native C/C++ library stderr (e.g. FFmpeg HEVC/RTSP decoder warnings)
os.environ["OPENCV_FFMPEG_LOG_LEVEL"] = "quiet"
os.environ["OPENCV_LOG_LEVEL"] = "OFF"



import cv2

if hasattr(cv2, 'utils') and hasattr(cv2.utils, 'logging'):
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        pass
from django.db import OperationalError
from django.db.models import F
from django.utils import timezone

from ..models import Camera, PersonProfile, SystemAlert, Video, VisitProfile, Visitor
from .clothing_service import get_clothing_service
from .gender_service import get_gender_service
from .video_utils import enqueue_transcode_to_h264, shutdown_transcode_queue, transcode_to_h264

# Auto-restart configuration for the camera worker loop.
_RESTART_MIN_DELAY = 5.0
_RESTART_MAX_DELAY = 300.0
_RESTART_BACKOFF_FACTOR = 2.0

# Seconds a person must be continuously tracked before being counted as a visitor.
_VISIT_SECONDS = 60
# Seconds to keep the "counted" highlight visible on screen.
_COUNT_HIGHLIGHT_SECONDS = 3
# Seconds without a detection before a tracked person is forgotten.
_TRACK_STALE_SECONDS = 30

# Person box validation thresholds to reduce false-positive YOLO detections.
_MIN_BOX_AREA_RATIO = 0.0005  # box must cover at least 0.05% of the frame
_MIN_BOX_WIDTH = 15
_MIN_BOX_HEIGHT = 20
_MIN_PERSON_ASPECT = 0.30
_MAX_PERSON_ASPECT = 4.0
_MIN_VISIBLE_KEYPOINTS = 3


def get_ai_hardware_info():
    """Get diagnostic information on PyTorch CUDA acceleration and OpenCV hardware capabilities."""
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_available else 'CPU'
        cuda_version = torch.version.cuda if cuda_available else 'N/A'
        torch_ver = torch.__version__
    except Exception:
        cuda_available = False
        device_name = 'CPU'
        cuda_version = 'N/A'
        torch_ver = 'N/A'

    opencv_cuda = 0
    if hasattr(cv2, 'cuda'):
        try:
            opencv_cuda = cv2.cuda.getCudaEnabledDeviceCount()
        except Exception:
            opencv_cuda = 0

    return {
        'cuda_available': cuda_available,
        'device_name': device_name,
        'cuda_version': cuda_version,
        'torch_version': torch_ver,
        'opencv_version': cv2.__version__,
        'opencv_cuda_devices': opencv_cuda,
        'acceleration_mode': f'GPU CUDA ({device_name})' if cuda_available else 'CPU (Software)',
    }


class CameraService:
    _instances = {}

    def __new__(cls, camera_id):
        if camera_id not in cls._instances:
            cls._instances[camera_id] = super().__new__(cls)
        return cls._instances[camera_id]

    def __init__(self, camera_id):
        if getattr(self, '_service_initialized', False):
            return
        self._service_initialized = True
        self.camera_id = camera_id
        self._thread = None
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._capture = None
        self._connected = False
        self._recording = False
        self._writer = None
        self._video_dir = None
        self._segment_start = None
        self._segment_path = None
        self._segment_count = 0
        self._trackers = {}
        self._detector_ready = False
        self._model = None
        self._pose_model = None
        self._initialized = False
        self._last_error = ''
        self._frame_count = 0
        self._latest_pose_results = None
        self._active_video = None
        self._active_video_id = None
        self._restart_timer = None
        self._restart_delay = _RESTART_MIN_DELAY
        self._lock = threading.Lock()
        self._gender_service = get_gender_service()
        self._clothing_service = get_clothing_service()
        self._tripwire_in_count = 0
        self._tripwire_out_count = 0
        self._heatmap_matrix = None

        # GPU acceleration configuration
        self._device = 'cpu'
        self._half = False

        # Real-time RTSP grabber & pre-encoded JPEG caching
        self._latest_jpeg = None
        self._latest_raw_frame = None
        self._raw_frame_lock = threading.Lock()
        self._grabber_thread = None
        self._grabber_stop_event = threading.Event()
        self._latest_results = None

    def initialize(self):
        if self._initialized:
            return True
        try:
            import torch
            from ultralytics import YOLO

            if torch.cuda.is_available():
                self._device = 'cuda:0'
                self._half = True
            else:
                self._device = 'cpu'
                self._half = False

            self._model = YOLO('yolov8n.pt')
            try:
                self._pose_model = YOLO('yolov8n-pose.pt')
            except Exception:
                self._pose_model = None
            self._detector_ready = True
        except Exception as exc:
            self._last_error = str(exc)
            self._detector_ready = False

        # Begin downloading the gender-classification models in the background.
        # The models (~24 MB total) are cached under MEDIA_ROOT/models, so this
        # only blocks the very first startup.
        try:
            threading.Thread(
                target=lambda: getattr(self._gender_service, 'ready'),
                name=f'gender-init-{self.camera_id}',
                daemon=True,
            ).start()
        except Exception:
            pass

        self._initialized = True
        return True

    def _resolve_source(self, camera):
        source = (camera.source or '').strip()
        if source:
            return source
        if camera.source_type in {'built_in_webcam', 'usb_webcam'}:
            return '0'
        raise RuntimeError('Camera source is empty. Please configure a valid camera source.')

    def _open_capture(self, camera):
        source = self._resolve_source(camera)

        capture = None
        source_int = None

        try:
            source_int = int(source)
        except (ValueError, TypeError):
            source_int = None

        if camera.source_type in ('rtsp', 'http_mjpeg') or (isinstance(source, str) and source.startswith(('rtsp://', 'rtsps://', 'http://', 'https://'))):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;102400|max_delay;500000|fflags;nobuffer+fastseek+flush_packets|flags;low_delay|framedrop|stimeout;3000000"
            os.environ["OPENCV_FFMPEG_LOG_LEVEL"] = "-8"
            os.environ["OPENCV_LOG_LEVEL"] = "OFF"


        backends_to_try = []
        if os.name == 'nt' and camera.source_type in {'built_in_webcam', 'usb_webcam'} and source_int is not None:
            # Try DirectShow first on Windows; MSMF often prints async sample error warnings (-1072873821)
            backends_to_try = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        else:
            backends_to_try = [cv2.CAP_ANY]

        if hasattr(cv2.utils.logging, 'setLogLevel'):
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
        for backend in backends_to_try:
            capture = None
            try:
                if source_int is not None:
                    capture = cv2.VideoCapture(source_int, backend)
                else:
                    capture = cv2.VideoCapture(source)

                if capture and capture.isOpened():
                    try:
                        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass

                    # For HEVC / RTSP streams, retry reading for up to 1.5 seconds to allow
                    # keyframes and VPS/SPS/PPS header parameters to arrive.
                    ok = False
                    frame = None
                    for attempt in range(15):
                        try:
                            ok, frame = capture.read()
                        except Exception:
                            ok = False
                            frame = None
                        if ok and frame is not None:
                            break
                        time.sleep(0.1)

                    if ok and frame is not None:
                        self._capture = capture
                        self._connected = True
                        with self._frame_lock:
                            self._latest_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        with self._raw_frame_lock:
                            self._latest_raw_frame = frame
                        self._start_grabber()
                        return capture

                    try:
                        capture.release()
                    except Exception:
                        pass
                    capture = None
            except Exception:
                if capture is not None:
                    try:
                        capture.release()
                    except Exception:
                        pass
                    capture = None
            time.sleep(0.2)

        if camera.source_type == 'rtsp' or (isinstance(source, str) and source.startswith(('rtsp://', 'rtsps://'))):
            raise RuntimeError(
                f"RTSP stream is not connected to '{source}'. Please verify camera power, network connectivity, RTSP port, credentials, and channel path."
            )
        elif camera.source_type == 'http_mjpeg' or (isinstance(source, str) and source.startswith(('http://', 'https://'))):
            raise RuntimeError(
                f"HTTP/MJPEG stream is not connected to '{source}'. Please verify the stream URL and network connectivity."
            )
        elif camera.source_type == 'usb_webcam':
            raise RuntimeError(
                f"USB webcam device (index {source or '1'}) is not connected or is in use by another app. Please check the USB connection."
            )
        elif camera.source_type == 'built_in_webcam':
            raise RuntimeError(
                f"Built-in webcam (index {source or '0'}) is unavailable. Please verify device permissions and camera privacy settings."
            )
        else:
            raise RuntimeError(
                f"Media source '{source}' could not be opened. Please verify the file path or stream address."
            )

    def _start_grabber(self):
        self._stop_grabber()
        self._grabber_stop_event.clear()
        self._grabber_thread = threading.Thread(target=self._grabber_loop, name=f'grabber-{self.camera_id}', daemon=True)
        self._grabber_thread.start()

    def _stop_grabber(self):
        self._grabber_stop_event.set()
        if self._grabber_thread and self._grabber_thread.is_alive():
            self._grabber_thread.join(timeout=1.0)
        self._grabber_thread = None

    def _grabber_loop(self):
        while not self._grabber_stop_event.is_set():
            cap = self._capture
            if cap is None or not cap.isOpened():
                time.sleep(0.01)
                continue
            try:
                ok, frame = cap.read()
                if ok and frame is not None:
                    with self._raw_frame_lock:
                        self._latest_raw_frame = frame
                else:
                    time.sleep(0.005)
            except Exception:
                time.sleep(0.01)

    def _close_capture(self):
        self._stop_grabber()
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
        self._capture = None
        self._connected = False
        with self._raw_frame_lock:
            self._latest_raw_frame = None

    def _save_camera_state(self, camera, *, status=None, last_error=None, updated_at=None):
        try:
            camera.refresh_from_db()
        except OperationalError:
            return False

        if status is not None:
            camera.status = status
        if last_error is not None:
            camera.last_error = last_error
        if updated_at is None:
            updated_at = timezone.now()
        camera.updated_at = updated_at
        try:
            camera.save(update_fields=['status', 'last_error', 'updated_at'])
        except OperationalError:
            return False
        if status in ('offline', 'error'):
            self._create_offline_alert(camera)
        return True

    def _cancel_restart(self):
        with self._lock:
            if self._restart_timer is not None:
                self._restart_timer.cancel()
                self._restart_timer = None

    def _schedule_restart(self):
        """If the camera is enabled, schedule an automatic restart with exponential backoff."""
        try:
            camera = Camera.objects.get(pk=self.camera_id)
        except OperationalError:
            # Database may be locked/closed (e.g. during tests). Do not crash the thread.
            return
        except Exception:
            return
        if not camera.enabled:
            return
        with self._lock:
            if self._restart_timer is not None:
                return

            def restart():
                with self._lock:
                    self._restart_timer = None
                self.start()

            self._restart_timer = threading.Timer(self._restart_delay, restart)
            self._restart_timer.daemon = True
            self._restart_timer.start()
            self._restart_delay = min(self._restart_delay * _RESTART_BACKOFF_FACTOR, _RESTART_MAX_DELAY)

    def start(self):
        if self._thread and self._thread.is_alive():
            return True
        try:
            camera = Camera.objects.get(pk=self.camera_id)
        except (OperationalError, Camera.DoesNotExist):
            self._schedule_restart()
            return False
        if camera.enabled is False:
            self._save_camera_state(camera, status='offline')
            return False

        self._cancel_restart()
        self._restart_delay = _RESTART_MIN_DELAY
        self.initialize()
        self._stop_event.clear()
        self._close_capture()
        self._close_writer()
        try:
            self._open_capture(camera)
        except Exception as exc:
            self._connected = False
            self._last_error = str(exc)
            self._save_camera_state(camera, status='error', last_error=self._last_error)
            self._schedule_restart()
            return False

        self._save_camera_state(camera, status='online', last_error='')
        self._thread = threading.Thread(target=self._run, args=(camera,), daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._cancel_restart()
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._close_capture()
        self._close_writer()
        camera = Camera.objects.get(pk=self.camera_id)
        self._save_camera_state(camera, status='offline')

    def _close_writer(self):
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception:
                pass
            self._writer = None
        if self._active_video is not None:
            self._finalize_active_video()
        self._recording = False

    def _finalize_active_video(self):
        if self._active_video is None:
            return

        if self._active_video.frames_count == 0:
            if os.path.exists(self._active_video.file_path):
                try:
                    os.remove(self._active_video.file_path)
                except OSError:
                    pass
            try:
                self._active_video.delete()
            except OperationalError:
                pass
            self._active_video = None
            self._active_video_id = None
            return

        elapsed = max(1, int(time.time() - self._segment_start)) if self._segment_start is not None else 0
        enqueue_transcode_to_h264(self._active_video.file_path)
        self._active_video.duration_seconds = elapsed
        self._active_video.ready = True
        try:
            self._active_video.save(update_fields=['duration_seconds', 'frames_count', 'ready'])
        except OperationalError:
            pass
        self._active_video = None
        self._active_video_id = None

    def _ensure_video_directory(self, camera):
        base_dir = os.path.join(settings.MEDIA_ROOT, 'videos', camera.slug)
        os.makedirs(base_dir, exist_ok=True)
        self._video_dir = base_dir
        return base_dir

    def _start_recording(self, camera):
        self._ensure_video_directory(camera)
        timestamp = timezone.now().strftime('%H-%M-%S')
        self._segment_path = os.path.join(self._video_dir, f'{timestamp}.mp4')
        self._segment_start = time.time()
        self._segment_count += 1
        self._recording = True
        try:
            self._active_video = Video.objects.create(
                camera=camera,
                file_path=self._segment_path,
                segment_start=timezone.now(),
                duration_seconds=0,
                frames_count=0,
                ready=False,
            )
        except OperationalError:
            self._active_video = None
        self._active_video_id = self._active_video.pk if self._active_video is not None else None
        return self._segment_path

    def _rotate_recording(self, camera):
        # Writer must be released (flushing the moov atom) before ffmpeg transcodes the file.
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception:
                pass
        self._writer = None
        self._finalize_active_video()
        self._recording = False
        if self._latest_frame is None:
            return
        self._start_recording(camera)
        w, h = self._latest_frame.shape[1], self._latest_frame.shape[0]
        self._writer_target_size = (w, h)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        try:
            self._writer = cv2.VideoWriter(self._segment_path, fourcc, 20.0, (w, h))
        except Exception:
            self._writer = None
        if self._writer is None or not self._writer.isOpened():
            self._writer = None
            self._recording = False

    def _write_frame(self, frame, camera):
        if frame is None or frame.size == 0:
            return
        h, w = frame.shape[:2]
        if self._writer is None:
            self._start_recording(camera)
            self._writer_target_size = (w, h)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            try:
                self._writer = cv2.VideoWriter(self._segment_path, fourcc, 20.0, (w, h))
            except Exception:
                self._writer = None
            if self._writer is None or not self._writer.isOpened():
                if self._active_video is not None:
                    try:
                        self._active_video.delete()
                    except Exception:
                        pass
                    self._active_video = None
                    self._active_video_id = None
                self._writer = None
                self._recording = False
                return

        target_size = getattr(self, '_writer_target_size', (w, h))
        if target_size and (w, h) != (target_size[0], target_size[1]):
            frame = cv2.resize(frame, target_size)

        try:
            self._writer.write(frame)
        except Exception:
            pass

        if self._active_video is not None:
            self._active_video.frames_count += 1
            if self._segment_start is not None:
                self._active_video.duration_seconds = int(time.time() - self._segment_start)
            if self._active_video.frames_count % 10 == 0:
                try:
                    self._active_video.save(update_fields=['frames_count', 'duration_seconds'])
                except OperationalError:
                    pass
        if self._segment_start is not None and time.time() - self._segment_start >= 30:
            self._rotate_recording(camera)

    def _cleanup_stale_tracks(self):
        """Forget tracks that have not been seen recently so counts stay accurate."""
        now = timezone.now()
        stale = [
            track_id
            for track_id, state in self._trackers.items()
            if (now - state['last_seen']).total_seconds() > _TRACK_STALE_SECONDS
        ]
        for track_id in stale:
            del self._trackers[track_id]

    def _validate_person_box(self, frame, x1, y1, x2, y2):
        """Return True if a person-shaped box actually looks human.

        Reject non-human false positives (small clutter, chairs, bags) using size
        and aspect ratio bounds. Pose estimation is used if available to verify
        human structure, but partial crops (seated/close-up webcams) are accepted.
        """
        h, w = frame.shape[:2]
        box_w = x2 - x1
        box_h = y2 - y1
        area = box_w * box_h
        frame_area = h * w

        if frame_area <= 0:
            return False
        if area < _MIN_BOX_AREA_RATIO * frame_area:
            return False
        if box_w < _MIN_BOX_WIDTH or box_h < _MIN_BOX_HEIGHT:
            return False

        aspect = box_h / max(1, box_w)
        if aspect < _MIN_PERSON_ASPECT or aspect > _MAX_PERSON_ASPECT:
            return False

        # Verification using pose keypoints if pose results match this box
        if self._latest_pose_results is not None:
            person_pose = self._select_pose_for_box(self._latest_pose_results, x1, y1, x2, y2)
            if person_pose is not None:
                try:
                    conf = person_pose.keypoints.conf[0].cpu().numpy()
                    visible = sum(1 for c in conf if c >= 0.3)
                    if visible >= 3:
                        return True
                except Exception:
                    pass

        # Fallback to valid aspect ratio for seated / close-up webcam views
        return True

    @staticmethod
    def _get_zone_rect(camera, frame_w, frame_h):
        """Return absolute pixel bounds (zx1, zy1, zx2, zy2) for the camera counting zone."""
        if not getattr(camera, 'zone_enabled', True):
            return 0, 0, frame_w, frame_h
        xmin = getattr(camera, 'zone_x_min', 0)
        ymin = getattr(camera, 'zone_y_min', 0)
        xmax = getattr(camera, 'zone_x_max', 100)
        ymax = getattr(camera, 'zone_y_max', 100)

        # Fallback to full frame area only if bounds are collapsed/invalid (0 width/height)
        if xmax <= xmin or ymax <= ymin:
            xmin, ymin, xmax, ymax = 0, 0, 100, 100

        xmin = max(0, min(100, xmin))
        ymin = max(0, min(100, ymin))
        xmax = max(xmin + 1, min(100, xmax))
        ymax = max(ymin + 1, min(100, ymax))
        zx1 = int(frame_w * xmin / 100.0)
        zy1 = int(frame_h * ymin / 100.0)
        zx2 = int(frame_w * xmax / 100.0)
        zy2 = int(frame_h * ymax / 100.0)
        return zx1, zy1, zx2, zy2

    @staticmethod
    def _is_box_in_zone(x1, y1, x2, y2, zx1, zy1, zx2, zy2):
        """Check if person is inside active zone using bounding box intersection or keypoints."""
        # 1. Bounding box intersection check (any spatial overlap with zone)
        ix1 = max(x1, zx1)
        iy1 = max(y1, zy1)
        ix2 = min(x2, zx2)
        iy2 = min(y2, zy2)
        if ix2 > ix1 and iy2 > iy1:
            inter_area = (ix2 - ix1) * (iy2 - iy1)
            if inter_area > 0:
                return True

        # 2. Center point check
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
            return True

        return False

    @staticmethod
    def _ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    @classmethod
    def _line_intersect(cls, A, B, C, D):
        """Return True if line segment AB intersects line segment CD."""
        return cls._ccw(A, C, D) != cls._ccw(B, C, D) and cls._ccw(A, B, C) != cls._ccw(A, B, D)

    def _draw_person_overlay(self, frame, track_id, x1, y1, x2, y2, camera=None):
        """Draw a bounding box and zone counting label above the person's head."""
        state = self._trackers.get(track_id)
        if state is None:
            return

        now = timezone.now()
        in_zone = state.get('in_zone', False)
        zone_counted = state.get('zone_counted', False)
        was_outside = state.get('was_outside_before', False)
        stage = state.get('zone_session_stage', 'outside')
        zone_elapsed = int((now - state['zone_entry_time']).total_seconds()) if (in_zone and state.get('zone_entry_time')) else 0

        zone_enabled = getattr(camera, 'zone_enabled', True) if camera else True
        if not zone_enabled:
            color = (0, 255, 0)
            thickness = 2
            count_num = state.get('person_visit_count') or state.get('visit_session_count')
            if count_num:
                label = f'Person #{track_id} (Count {count_num})'
            else:
                label = f'Person #{track_id}'
        elif not in_zone:
            color = (180, 140, 90)
            thickness = 2
            if stage == 'leaving':
                label = 'Leaving Zone'
            else:
                label = 'Outside Zone'
        elif zone_counted:
            count_num = state.get('person_visit_count') or state.get('visit_session_count') or 1
            color = (0, 255, 0)
            thickness = 2
            if state.get('highlight_until') and now < state['highlight_until']:
                color = (0, 255, 255)
                thickness = 3
                label = f'COUNTED! (Visit #{count_num})'
            else:
                label = f'In Zone (Already Counted)'
        else:
            thickness = 2
            if zone_elapsed >= 50:
                color = (0, 165, 255)
            else:
                color = (0, 255, 0)

            if stage == 're_entering':
                label = f'Re-entered Zone ({zone_elapsed}s/60s)'
            else:
                label = f'In Zone ({zone_elapsed}s/60s)'

        # Bounding box with sleek 2px outline
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Check for VIP / Blacklist / Custom Name tagging
        person_type = state.get('person_type', 'regular')
        person_name = state.get('person_name', '')

        tag_prefix = ''
        if person_type == 'vip':
            tag_prefix = '[VIP] '
        elif person_type == 'blacklist':
            tag_prefix = '[ALERT: BLACKLIST] '

        if person_name:
            label = f'{tag_prefix}{person_name} ({label})'
        elif tag_prefix:
            label = f'{tag_prefix}{label}'

        # Label background pill and crisp text above the head.
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
        label_w, label_h = label_size[0], label_size[1]
        
        # Stagger y-position slightly based on track_id to avoid multi-label collision
        y_offset = (track_id % 3) * 6
        label_x = max(2, min(frame.shape[1] - label_w - 6, x1))
        label_y = max(label_h + 8, y1 - 6 - y_offset)

        # Dark glass pill background
        badge_bg = (0, 0, 180) if person_type == 'blacklist' else ((255, 140, 0) if person_type == 'vip' else (24, 28, 36))
        text_color = (255, 255, 255) if person_type in ('vip', 'blacklist') else color

        cv2.rectangle(
            frame,
            (label_x - 2, label_y - label_h - 4),
            (label_x + label_w + 6, label_y + 4),
            badge_bg,
            -1,
        )
        cv2.rectangle(
            frame,
            (label_x - 2, label_y - label_h - 4),
            (label_x + label_w + 6, label_y + 4),
            color,
            1,
        )
        cv2.putText(
            frame,
            label,
            (label_x + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            text_color,
            2,
        )

    def _run(self, camera):
        reopen_failures = 0
        try:
            self._open_capture(camera)
            self._start_recording(camera)
            while not self._stop_event.is_set():
                if self._capture is None or not self._capture.isOpened():
                    self._close_capture()
                    self._close_writer()
                    try:
                        self._open_capture(camera)
                        reopen_failures = 0
                    except Exception:
                        reopen_failures += 1
                        time.sleep(min(0.5 * reopen_failures, 5.0))
                        continue

                frame = None
                with self._raw_frame_lock:
                    if self._latest_raw_frame is not None:
                        frame = self._latest_raw_frame
                        self._latest_raw_frame = None

                if frame is None:
                    if self._grabber_thread is None or not self._grabber_thread.is_alive():
                        ok, frame = self._capture.read()
                        if not ok or frame is None:
                            self._close_capture()
                            self._close_writer()
                            try:
                                self._open_capture(camera)
                                reopen_failures = 0
                            except Exception:
                                reopen_failures += 1
                            time.sleep(min(0.5 * reopen_failures, 5.0))
                            continue
                    else:
                        time.sleep(0.005)
                        continue

                # Detect people, update counters, and burn the overlays into the frame
                # so both the live stream and the recording show elapsed seconds / counts.
                annotated = self._process_frame(frame, camera)

                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                ok_jpg, buf = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                jpg_bytes = buf.tobytes() if ok_jpg else None

                with self._frame_lock:
                    self._latest_frame = rgb
                    if jpg_bytes:
                        self._latest_jpeg = jpg_bytes

                if self._frame_count % 2 == 0 and jpg_bytes:
                    try:
                        from django.core.cache import cache
                        from django.db import close_old_connections
                        close_old_connections()
                        cache.set(f'vms:live_frame:{camera.pk}', jpg_bytes, timeout=10)
                        cache.set(f'vms:camera_state:{camera.pk}', self.get_camera_state(camera), timeout=30)
                        if camera.status != 'online' or camera.last_error:
                            self._save_camera_state(camera, status='online', last_error='')
                    except Exception:
                        pass

                self._write_frame(annotated, camera)
                time.sleep(0.005)
        except Exception as exc:
            self._last_error = str(exc)
            self._save_camera_state(camera, status='error', last_error=self._last_error)
            self._close_capture()
            self._close_writer()
            self._schedule_restart()
        else:
            self._save_camera_state(camera, status='offline')

    def _process_frame(self, frame, camera):
        """Run person tracking, update visitor state, draw overlays, and return annotated frame."""
        if not self._detector_ready or self._model is None:
            return frame

        try:
            self._frame_count += 1
            if self._frame_count % 10 == 0:
                try:
                    camera.refresh_from_db()
                except Exception:
                    pass

            model_kwargs = {
                'persist': True,
                'tracker': 'bytetrack.yaml',
                'conf': 0.25,
                'iou': 0.45,
                'classes': [0],
                'imgsz': 416,
                'stream': False,
                'verbose': False,
                'device': self._device,
            }
            if self._half:
                model_kwargs['half'] = True

            # Run pose estimation every 6 frames on CPU to keep CPU load low on VPS
            if self._pose_model is not None and self._frame_count % 6 == 0:
                try:
                    pose_kwargs = {'verbose': False, 'device': self._device, 'imgsz': 384}
                    if self._half:
                        pose_kwargs['half'] = True
                    pose_results = self._pose_model(frame, **pose_kwargs)
                    self._latest_pose_results = pose_results
                except Exception:
                    pass

            # Run YOLO detection every 2 frames to cut CPU usage by 50% while preserving real-time tracking
            if self._frame_count % 2 == 0 or self._latest_results is None:
                self._latest_results = self._model.track(frame, **model_kwargs)
            results = self._latest_results
            annotated = frame.copy()
            h, w = frame.shape[:2]

            # 1. Render Counting Zone Overlay
            if getattr(camera, 'zone_enabled', True):
                zx1, zy1, zx2, zy2 = self._get_zone_rect(camera, w, h)
                
                # Soft translucent fill for the zone area
                overlay = annotated.copy()
                cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), (0, 255, 255), -1)
                cv2.addWeighted(overlay, 0.12, annotated, 0.88, 0, annotated)

                # Bright yellow boundary rectangle
                cv2.rectangle(annotated, (zx1, zy1), (zx2, zy2), (0, 255, 255), 2)

                # Bold corner brackets for high-visibility zone definition
                corner_len = max(15, min(30, (zx2 - zx1) // 6, (zy2 - zy1) // 6))
                # Top-Left
                cv2.line(annotated, (zx1, zy1), (zx1 + corner_len, zy1), (0, 255, 255), 4)
                cv2.line(annotated, (zx1, zy1), (zx1, zy1 + corner_len), (0, 255, 255), 4)
                # Top-Right
                cv2.line(annotated, (zx2, zy1), (zx2 - corner_len, zy1), (0, 255, 255), 4)
                cv2.line(annotated, (zx2, zy1), (zx2, zy1 + corner_len), (0, 255, 255), 4)
                # Bottom-Left
                cv2.line(annotated, (zx1, zy2), (zx1 + corner_len, zy2), (0, 255, 255), 4)
                cv2.line(annotated, (zx1, zy2), (zx1, zy2 - corner_len), (0, 255, 255), 4)
                # Bottom-Right
                cv2.line(annotated, (zx2, zy2), (zx2 - corner_len, zy2), (0, 255, 255), 4)
                cv2.line(annotated, (zx2, zy2), (zx2, zy2 - corner_len), (0, 255, 255), 4)

                # Counting Zone Banner Badge
                z_text = 'COUNTING ZONE AREA (60s)'
                (zw, zh), _ = cv2.getTextSize(z_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                badge_x = max(2, zx1)
                badge_y = max(zh + 6, zy1)
                cv2.rectangle(annotated, (badge_x, badge_y - zh - 6), (badge_x + zw + 14, badge_y + 4), (0, 160, 160), -1)
                cv2.putText(
                    annotated,
                    z_text,
                    (badge_x + 6, badge_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

            # 2. Render Tripwire Line Crossing Overlay if enabled
            if getattr(camera, 'tripwire_enabled', True):
                tx1 = int(w * getattr(camera, 'tripwire_x1', 10) / 100.0)
                ty1 = int(h * getattr(camera, 'tripwire_y1', 50) / 100.0)
                tx2 = int(w * getattr(camera, 'tripwire_x2', 90) / 100.0)
                ty2 = int(h * getattr(camera, 'tripwire_y2', 50) / 100.0)

                # Draw magenta/red tripwire line with glowing endpoints
                cv2.line(annotated, (tx1, ty1), (tx2, ty2), (0, 0, 255), 3)
                cv2.circle(annotated, (tx1, ty1), 6, (0, 255, 255), -1)
                cv2.circle(annotated, (tx1, ty1), 6, (0, 0, 255), 2)
                cv2.circle(annotated, (tx2, ty2), 6, (0, 255, 255), -1)
                cv2.circle(annotated, (tx2, ty2), 6, (0, 0, 255), 2)

                # Tripwire Banner Pill anchored cleanly at start point
                trip_text = f'TRIPWIRE LINE [IN: {self._tripwire_in_count} | OUT: {self._tripwire_out_count}]'
                (tw, th), _ = cv2.getTextSize(trip_text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
                lbl_x = max(4, min(w - tw - 10, tx1))
                lbl_y = max(th + 8, ty1 - 10) if ty1 > th + 15 else (ty1 + th + 14)

                cv2.rectangle(annotated, (lbl_x - 2, lbl_y - th - 4), (lbl_x + tw + 6, lbl_y + 4), (20, 20, 120), -1)
                cv2.rectangle(annotated, (lbl_x - 2, lbl_y - th - 4), (lbl_x + tw + 6, lbl_y + 4), (0, 0, 255), 1)
                cv2.putText(
                    annotated,
                    trip_text,
                    (lbl_x + 2, lbl_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (255, 255, 255),
                    2,
                )

            candidate_boxes = []
            for result in results:
                boxes = getattr(result, 'boxes', None)
                if boxes is None:
                    continue
                track_ids = getattr(boxes, 'id', None)
                if track_ids is None:
                    continue
                for track_id, box in zip(track_ids.tolist(), boxes.xyxy):
                    track_id = int(track_id)
                    x1, y1, x2, y2 = map(int, box.tolist())
                    if self._validate_person_box(frame, x1, y1, x2, y2):
                        candidate_boxes.append((track_id, (x1, y1, x2, y2)))

            # Suppress sub-boxes / duplicate overlapping detections on the same physical person
            filtered_boxes = self._suppress_overlapping_boxes(candidate_boxes)

            seen_track_ids = set()
            for track_id, (x1, y1, x2, y2) in filtered_boxes:
                if track_id in seen_track_ids:
                    continue
                seen_track_ids.add(track_id)
                self._update_track_state(camera, track_id, x1, y1, x2, y2, frame)
                self._draw_person_overlay(annotated, track_id, x1, y1, x2, y2, camera)

            # Periodically remove people who left the frame.
            self._cleanup_stale_tracks()
            return annotated
        except Exception:
            return frame

    @staticmethod
    def _suppress_overlapping_boxes(boxes_with_ids, iou_threshold=0.45):
        """Suppress smaller duplicate boxes that heavily overlap larger boxes for the same person."""
        if not boxes_with_ids:
            return []

        sorted_boxes = sorted(boxes_with_ids, key=lambda item: (item[1][2] - item[1][0]) * (item[1][3] - item[1][1]), reverse=True)
        keep = []

        for tid, (x1, y1, x2, y2) in sorted_boxes:
            area = (x2 - x1) * (y2 - y1)
            should_keep = True
            for ktid, (kx1, ky1, kx2, ky2) in keep:
                karea = (kx2 - kx1) * (ky2 - ky1)
                ix1, iy1 = max(x1, kx1), max(y1, ky1)
                ix2, iy2 = min(x2, kx2), min(y2, ky2)
                if ix2 > ix1 and iy2 > iy1:
                    iarea = (ix2 - ix1) * (iy2 - iy1)
                    iou = iarea / float(area + karea - iarea)
                    containment = iarea / float(min(area, karea))
                    if iou > iou_threshold or containment > 0.60:
                        should_keep = False
                        break
            if should_keep:
                keep.append((tid, (x1, y1, x2, y2)))
        return keep

    def _cleanup_stale_tracks(self, timeout=15.0):
        """Remove tracker states for people who have not been seen for `timeout` seconds."""
        now = timezone.now()
        stale_ids = [
            tid for tid, state in self._trackers.items()
            if (now - state.get('last_seen', now)).total_seconds() > timeout
        ]
        for tid in stale_ids:
            del self._trackers[tid]

    def _update_track_state(self, camera, track_id, x1, y1, x2, y2, frame):
        now = timezone.now()
        h, w = frame.shape[:2]
        zx1, zy1, zx2, zy2 = self._get_zone_rect(camera, w, h)
        is_in_zone_now = self._is_box_in_zone(x1, y1, x2, y2, zx1, zy1, zx2, zy2)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        state = self._trackers.get(track_id)
        if state is None:
            embedding = self._extract_embedding(frame, x1, y1, x2, y2)
            matched_profile = self._find_matching_person(camera, embedding) if embedding else None
            person_type = matched_profile.person_type if matched_profile else 'regular'
            person_name = matched_profile.name if matched_profile else ''

            # Check if an existing active tracker for this camera overlaps or matches this person inside the zone
            inherited_zone_counted = False
            inherited_zone_entry_time = now if is_in_zone_now else None
            inherited_zone_exit_time = None
            inherited_was_outside = False
            inherited_person_visit_count = None
            inherited_visitor_id = None

            overlap_track_id = None
            for tid, tstate in list(self._trackers.items()):
                if tid == track_id:
                    continue
                # Increased spatial overlap tolerance (35% box width/height) to maintain continuity during seated wiggles/occlusions
                if (matched_profile and tstate.get('person_profile_id') == matched_profile.pk) or \
                   (tstate.get('prev_cx') is not None and abs(tstate['prev_cx'] - cx) < w * 0.35 and abs(tstate['prev_cy'] - cy) < h * 0.35):
                    overlap_track_id = tid
                    inherited_zone_counted = tstate.get('zone_counted', False) or tstate.get('counted', False)
                    inherited_zone_entry_time = tstate.get('zone_entry_time') or (now if is_in_zone_now else None)
                    inherited_zone_exit_time = tstate.get('zone_exit_time')
                    inherited_was_outside = tstate.get('was_outside_before', False)
                    inherited_person_visit_count = tstate.get('person_visit_count')
                    inherited_visitor_id = tstate.get('visitor_id')
                    break

            if overlap_track_id is not None:
                del self._trackers[overlap_track_id]

            self._trackers[track_id] = {
                'first_seen': now,
                'last_seen': now,
                'in_zone': is_in_zone_now,
                'out_of_zone_frames': 0,
                'zone_entry_time': inherited_zone_entry_time,
                'zone_exit_time': inherited_zone_exit_time,
                'zone_counted': inherited_zone_counted,
                'was_outside_before': inherited_was_outside,
                'person_visit_count': inherited_person_visit_count,
                'counted': inherited_zone_counted,
                'visitor_id': inherited_visitor_id,
                'highlight_until': None,
                'required_seconds': _VISIT_SECONDS,
                'embedding': embedding,
                'person_profile_id': matched_profile.pk if matched_profile else None,
                'person_type': person_type,
                'person_name': person_name,
                'prev_cx': cx,
                'prev_cy': cy,
                'gender_votes': {'male': 0, 'female': 0, 'unknown': 0},
                'gender_last_sampled': None,
                'zone_session_stage': 're_entering' if inherited_was_outside else ('entering' if is_in_zone_now else 'outside'),
            }
            state = self._trackers[track_id]
        else:
            state['last_seen'] = now
            if 'in_zone' not in state:
                state['in_zone'] = is_in_zone_now
                state['out_of_zone_frames'] = 0
                state['zone_entry_time'] = state.get('first_seen', now) if is_in_zone_now else None
                state['zone_counted'] = state.get('counted', False)

        # Zone Entry / Exit / Re-entry State Machine
        if is_in_zone_now:
            state['out_of_zone_frames'] = 0
            if not state.get('in_zone', False):
                was_outside = state.get('was_outside_before', False)
                state['in_zone'] = True
                
                # Check if person was outside long enough (>10 seconds) to warrant a brand-new visit count
                exit_time = state.get('zone_exit_time')
                long_exit = exit_time is not None and (now - exit_time).total_seconds() >= 10.0

                if long_exit:
                    # Genuine re-entry after long absence -> start a new counting session
                    state['zone_entry_time'] = now
                    state['zone_counted'] = False
                    state['counted'] = False
                    state['zone_session_stage'] = 're_entering'
                else:
                    # Brief departure/flicker -> maintain existing counted state, do not restart timer
                    if state.get('zone_entry_time') is None:
                        state['zone_entry_time'] = now
                    state['zone_session_stage'] = 're_entering' if was_outside else 'entering'
        else:
            out_frames = state.get('out_of_zone_frames', 0) + 1
            state['out_of_zone_frames'] = out_frames
            # Require 5 consecutive frames outside before marking person as out-of-zone
            if state.get('in_zone', False) and out_frames >= 5:
                state['in_zone'] = False
                state['zone_counted'] = False
                state['zone_entry_time'] = None
                state['zone_exit_time'] = now
                state['was_outside_before'] = True
                state['zone_session_stage'] = 'leaving'

        # Tripwire Line Crossing Check
        if getattr(camera, 'tripwire_enabled', True):
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            prev_cx = state.get('prev_cx', cx)
            prev_cy = state.get('prev_cy', cy)

            tx1 = int(w * getattr(camera, 'tripwire_x1', 10) / 100.0)
            ty1 = int(h * getattr(camera, 'tripwire_y1', 50) / 100.0)
            tx2 = int(w * getattr(camera, 'tripwire_x2', 90) / 100.0)
            ty2 = int(h * getattr(camera, 'tripwire_y2', 50) / 100.0)

            p1 = (prev_cx, prev_cy)
            p2 = (cx, cy)
            t1 = (tx1, ty1)
            t2 = (tx2, ty2)

            if p1 != p2 and self._line_intersect(p1, p2, t1, t2):
                dx_trip = tx2 - tx1
                dy_trip = ty2 - ty1
                dx_move = cx - prev_cx
                dy_move = cy - prev_cy
                cross = dx_trip * dy_move - dy_trip * dx_move
                if cross > 0:
                    self._tripwire_in_count += 1
                    state['tripwire_direction'] = 'in'
                else:
                    self._tripwire_out_count += 1
                    state['tripwire_direction'] = 'out'

            state['prev_cx'] = cx
            state['prev_cy'] = cy

        # Sample gender a few times while the person is being tracked.
        total_elapsed = (now - state['first_seen']).total_seconds()
        if total_elapsed >= 5:
            last_sampled = state.get('gender_last_sampled')
            if last_sampled is None or (now - last_sampled).total_seconds() >= 3:
                gender = self._gender_service.detect_gender(frame, x1, y1, x2, y2)
                if gender == 'unknown':
                    gender = self._estimate_gender(self._latest_pose_results, x1, y1, x2, y2)
                votes = state.setdefault('gender_votes', {'male': 0, 'female': 0, 'unknown': 0})
                votes[gender] = votes.get(gender, 0) + 1
                state['gender_last_sampled'] = now

        # Calculate continuous time spent inside the zone for the current session
        zone_elapsed = 0.0
        if state['in_zone'] and state['zone_entry_time'] is not None:
            zone_elapsed = (now - state['zone_entry_time']).total_seconds()

        if state['in_zone'] and not state.get('zone_counted', False) and zone_elapsed >= state.get('required_seconds', _VISIT_SECONDS):
            state['zone_counted'] = True
            state['counted'] = True
            state['visit_session_count'] = state.get('visit_session_count', 0) + 1
            state['highlight_until'] = now + timezone.timedelta(seconds=_COUNT_HIGHLIGHT_SECONDS)

            defaults = {
                'entry_time': state['zone_entry_time'] or state['first_seen'],
                'counted_time': now,
                'dwell_time': int(zone_elapsed),
            }
            active_vid = self._get_valid_active_video()
            if active_vid is not None:
                defaults['video'] = active_vid

            visitor, _ = Visitor.objects.get_or_create(
                camera=camera,
                track_id=track_id,
                defaults=defaults,
            )
            visitor.entry_time = defaults['entry_time']
            visitor.counted_time = defaults['counted_time']
            visitor.dwell_time = defaults['dwell_time']
            if defaults.get('video') is not None:
                visitor.video = defaults['video']
            visitor.save()
            state['visitor_id'] = visitor.pk

            # Create the daily visit profile.
            self._create_visit_profile(camera, track_id, state, now, zone_elapsed, frame, x1, y1, x2, y2)

    def _create_visit_profile(self, camera, track_id, state, now, elapsed, frame, x1, y1, x2, y2):
        """Create a VisitProfile for this camera, assigning the next sequence number for the day."""
        from django.db import transaction

        today = now.date()
        pose_results = self._latest_pose_results

        # Resolve gender from accumulated face-based votes.  If we do not have
        # a confident estimate yet, try one fresh detection and finally fall
        # back to the pose-based heuristic.
        gender = self._resolve_gender_from_votes(state.get('gender_votes', {}))
        if gender == 'unknown':
            gender = self._gender_service.detect_gender(frame, x1, y1, x2, y2)
        if gender == 'unknown':
            gender = self._estimate_gender(pose_results, x1, y1, x2, y2)

        pose_label = self._estimate_pose_label(pose_results, x1, y1, x2, y2)
        liveness = self._estimate_liveness(pose_results, x1, y1, x2, y2)
        snapshot_path = self._save_snapshot(frame, x1, y1, x2, y2)

        # Classify clothing / attire using the person crop and resolved gender.
        attire_result = self._clothing_service.classify(frame, x1, y1, x2, y2, gender=gender)
        attire = attire_result.get('category', 'unknown')
        attire_attributes = attire_result.get('attributes', {})
        attire_label = attire_result.get('label', attire)

        with transaction.atomic():
            last_profile = (
                VisitProfile.objects.filter(camera=camera, date=today)
                .select_for_update()
                .order_by('-sequence_number')
                .first()
            )
            sequence_number = 1 if last_profile is None else last_profile.sequence_number + 1
            required_seconds = _VISIT_SECONDS

            person_profile_id = state.get('person_profile_id')
            person_profile = None
            if person_profile_id:
                try:
                    person_profile = PersonProfile.objects.select_for_update().get(pk=person_profile_id)
                except PersonProfile.DoesNotExist:
                    person_profile = None

            if person_profile is None:
                embedding = state.get('embedding') or self._extract_embedding(frame, x1, y1, x2, y2)
                person_profile = PersonProfile.objects.create(
                    camera=camera,
                    first_seen=state['first_seen'],
                    last_seen=now,
                    visit_count=1,
                    gender=gender,
                    attire=attire,
                    attire_attributes=attire_attributes,
                    embedding=embedding or [],
                )
            else:
                person_profile.last_seen = now
                person_profile.visit_count = F('visit_count') + 1
                if person_profile.gender == 'unknown' and gender != 'unknown':
                    person_profile.gender = gender
                # Keep the most confident / most recent attire observation.
                if attire != 'unknown':
                    person_profile.attire = attire
                    person_profile.attire_attributes = attire_attributes
                person_profile.save(update_fields=['last_seen', 'visit_count', 'gender', 'attire', 'attire_attributes'])
                person_profile.refresh_from_db()

            VisitProfile.objects.create(
                camera=camera,
                person_profile=person_profile,
                video=self._get_valid_active_video(),
                track_id=track_id,
                sequence_number=sequence_number,
                date=today,
                entry_time=state['first_seen'],
                counted_time=now,
                dwell_time=int(elapsed),
                required_seconds=required_seconds,
                gender=gender,
                attire=attire,
                attire_attributes=attire_attributes,
                pose=pose_label,
                liveness_score=liveness,
                snapshot_path=snapshot_path,
            )
            state['person_profile_id'] = person_profile.pk
            state['person_visit_count'] = person_profile.visit_count
            state['attire_label'] = attire_label
            state['attire_attributes'] = attire_attributes

    def _get_valid_active_video(self):
        if self._active_video is not None:
            try:
                if self._active_video.pk and Video.objects.filter(pk=self._active_video.pk).exists():
                    return self._active_video
            except Exception:
                pass
            self._active_video = None
            self._active_video_id = None
        return None

    def _create_offline_alert(self, camera):
        """Create an in-app alert when a camera goes offline/error."""
        if not camera.alert_on_offline:
            return
        # Avoid spamming alerts every second.
        recent = SystemAlert.objects.filter(
            camera=camera,
            title__icontains='offline',
            acknowledged=False,
        ).order_by('-created_at').first()
        if recent and (timezone.now() - recent.created_at).total_seconds() < 300:
            return
        SystemAlert.objects.create(
            camera=camera,
            title=f'{camera.name} is {camera.status}',
            message=f'The camera "{camera.name}" reported status "{camera.status}". Last error: {camera.last_error or "None"}',
        )

    def _extract_embedding(self, frame, x1, y1, x2, y2):
        """Create a simple color-histogram embedding from a person crop."""
        try:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                return None
            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, (64, 128))
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [8, 6], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            return hist.flatten().tolist()
        except Exception:
            return None

    @staticmethod
    def _compare_embeddings(a, b):
        """Cosine similarity between two embeddings."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _find_matching_person(self, camera, embedding):
        """Return the most similar PersonProfile for this camera, if any."""
        if not embedding:
            return None
        best_match = None
        best_score = 0.0
        for profile in PersonProfile.objects.filter(camera=camera).iterator():
            score = self._compare_embeddings(embedding, profile.embedding)
            if score > best_score and score >= 0.72:
                best_score = score
                best_match = profile
        return best_match

    @staticmethod
    def _resolve_gender_from_votes(votes):
        """Return the gender with the most accumulated votes."""
        if not votes:
            return 'unknown'
        male = votes.get('male', 0)
        female = votes.get('female', 0)
        if male > female:
            return 'male'
        if female > male:
            return 'female'
        return 'unknown'

    def _estimate_gender(self, pose_results, x1, y1, x2, y2):
        """Estimate gender from shoulder-to-hip ratio using pose keypoints."""
        try:
            person_pose = self._select_pose_for_box(pose_results, x1, y1, x2, y2)
            if person_pose is None:
                return 'unknown'
            kps = person_pose.keypoints.xy[0].cpu().numpy()
            conf = person_pose.keypoints.conf[0].cpu().numpy()
            if len(kps) < 13:
                return 'unknown'
            # COCO keypoints: 5=left_shoulder, 6=right_shoulder, 11=left_hip, 12=right_hip
            indices = [5, 6, 11, 12]
            if any(conf[i] < 0.3 for i in indices):
                return 'unknown'
            shoulder_width = abs(kps[5][0] - kps[6][0])
            hip_width = abs(kps[11][0] - kps[12][0])
            if hip_width == 0:
                return 'unknown'
            ratio = shoulder_width / hip_width
            if ratio > 1.35:
                return 'male'
            if ratio < 1.15:
                return 'female'
            return 'unknown'
        except Exception:
            return 'unknown'

    def _estimate_pose_label(self, pose_results, x1, y1, x2, y2):
        """Classify basic pose from keypoints."""
        try:
            person_pose = self._select_pose_for_box(pose_results, x1, y1, x2, y2)
            if person_pose is None:
                return 'unknown'
            kps = person_pose.keypoints.xy[0].cpu().numpy()
            conf = person_pose.keypoints.conf[0].cpu().numpy()
            if len(kps) < 13:
                return 'unknown'
            # COCO: nose=0, left_eye=1, right_eye=2, left_ear=3, right_ear=4,
            # left_shoulder=5, right_shoulder=6, left_elbow=7, right_elbow=8,
            # left_wrist=9, right_wrist=10, left_hip=11, right_hip=12
            ys = {i: kps[i][1] for i in range(len(kps)) if conf[i] >= 0.3}
            if not ys:
                return 'unknown'
            hip_y = min(ys.get(11, float('inf')), ys.get(12, float('inf')))
            knee_y = min(ys.get(13, float('inf')), ys.get(14, float('inf')))
            ankle_y = min(ys.get(15, float('inf')), ys.get(16, float('inf')))
            shoulder_y = min(ys.get(5, float('inf')), ys.get(6, float('inf')))
            wrist_y = min(ys.get(9, float('inf')), ys.get(10, float('inf')))
            # Lying down: shoulders and hips roughly same height.
            if hip_y != float('inf') and shoulder_y != float('inf') and abs(hip_y - shoulder_y) < 0.2 * (y2 - y1):
                return 'lying_down'
            # Sitting: knees not far below hips.
            if knee_y != float('inf') and hip_y != float('inf') and knee_y - hip_y < 0.3 * (y2 - y1):
                return 'sitting'
            # Hand raised: wrist above shoulder.
            if wrist_y != float('inf') and shoulder_y != float('inf') and wrist_y < shoulder_y:
                return 'hand_raised'
            # Standing: hips well above knees/ankles.
            if hip_y != float('inf') and ankle_y != float('inf') and hip_y < ankle_y:
                return 'standing'
            return 'unknown'
        except Exception:
            return 'unknown'

    def _estimate_liveness(self, pose_results, x1, y1, x2, y2):
        """Return a simple liveness score based on visible pose keypoints."""
        try:
            person_pose = self._select_pose_for_box(pose_results, x1, y1, x2, y2)
            if person_pose is None:
                return 0.5
            conf = person_pose.keypoints.conf[0].cpu().numpy()
            visible = sum(1 for c in conf if c >= 0.3)
            return min(0.95, 0.5 + visible * 0.03)
        except Exception:
            return 0.5

    def _select_pose_for_box(self, pose_results, x1, y1, x2, y2):
        """Return the pose result whose bounding box best overlaps the person box."""
        if pose_results is None:
            return None
        best = None
        best_iou = 0.0
        target_area = max(1, (x2 - x1) * (y2 - y1))
        for result in pose_results:
            boxes = getattr(result, 'boxes', None)
            if boxes is None:
                continue
            for box in boxes.xyxy:
                px1, py1, px2, py2 = map(int, box.tolist())
                inter_x1 = max(x1, px1)
                inter_y1 = max(y1, py1)
                inter_x2 = min(x2, px2)
                inter_y2 = min(y2, py2)
                inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                union = target_area + max(1, (px2 - px1) * (py2 - py1)) - inter
                iou = inter / union if union > 0 else 0
                if iou > best_iou:
                    best_iou = iou
                    best = result
            if best_iou > 0.5:
                break
        return best

    def _save_snapshot(self, frame, x1, y1, x2, y2):
        """Save a cropped image of the person as evidence."""
        try:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                return ''
            crop = frame[y1:y2, x1:x2]
            snapshot_dir = os.path.join(settings.MEDIA_ROOT, 'snapshots')
            os.makedirs(snapshot_dir, exist_ok=True)
            timestamp = timezone.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f'snapshot_{timestamp}.jpg'
            path = os.path.join(snapshot_dir, filename)
            cv2.imwrite(path, crop)
            return os.path.relpath(path, settings.MEDIA_ROOT).replace('\\', '/')
        except Exception:
            return ''

    def get_latest_frame(self):
        with self._frame_lock:
            return self._latest_frame

    def get_latest_jpeg(self):
        with self._frame_lock:
            return self._latest_jpeg

    def get_camera_state(self, camera):
        from django.core.cache import cache
        from django.db import close_old_connections
        close_old_connections()

        if not self._connected and not (self._thread and self._thread.is_alive()):
            try:
                cached_state = cache.get(f'vms:camera_state:{camera.pk}')
                if cached_state is not None:
                    return cached_state
            except Exception:
                pass

        recording_duration = 0
        if self._recording and self._segment_start is not None:
            recording_duration = int(time.time() - self._segment_start)
        elif self._active_video is not None:
            recording_duration = self._active_video.duration_seconds

        def fmt_duration(seconds):
            if seconds <= 0:
                return '00:00'
            minutes, secs = divmod(seconds, 60)
            return f'{minutes:02d}:{secs:02d}'

        # Count active persons currently inside the zone
        zone_occupancy = sum(1 for t in self._trackers.values() if t.get('in_zone', False))
        max_capacity = getattr(camera, 'queue_max_capacity', 5)
        overcrowded = (zone_occupancy > max_capacity) if getattr(camera, 'queue_alert_enabled', True) else False

        is_live = self._connected or bool(cache.get(f'vms:live_frame:{camera.pk}'))
        current_status = 'online' if is_live else camera.status
        current_error = '' if is_live else camera.last_error

        state = {
            'status': current_status,
            'connected': is_live,
            'recording': self._recording,
            'analytics': self._detector_ready or is_live,
            'active_tracks': len(self._trackers),
            'zone_occupancy': zone_occupancy,
            'queue_max_capacity': max_capacity,
            'queue_overcrowded': overcrowded,
            'tripwire_in': self._tripwire_in_count,
            'tripwire_out': self._tripwire_out_count,
            'visitor_count': Visitor.objects.filter(camera=camera, counted_time__date=timezone.now().date()).count(),
            'last_error': current_error,
            'motion_detected': False,
            'recording_duration': recording_duration,
            'recording_duration_display': fmt_duration(recording_duration),
        }

        try:
            cache.set(f'vms:camera_state:{camera.pk}', state, timeout=30)
        except Exception:
            pass

        return state

