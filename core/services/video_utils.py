"""Helpers to make OpenCV-written videos playable in HTML5 <video> tags.

cv2.VideoWriter with the 'mp4v' fourcc produces MPEG-4 Part 2 video streams.
OpenCV/ffmpeg can decode that fine (which is why local validation succeeds),
but no mainstream browser can decode it, so the resulting "video/mp4" files
fail to play. We re-encode to H.264 (with a front-loaded moov atom) using the
static ffmpeg binary shipped by imageio-ffmpeg.
"""
import logging
import os
import queue
import subprocess
import threading
import time

logger = logging.getLogger(__name__)

try:
    import imageio_ffmpeg
    FFMPEG_BINARY = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_BINARY = None


def transcode_to_h264(path):
    """Re-encode the video at `path` to H.264 in place. Returns True on success."""
    if not FFMPEG_BINARY or not os.path.exists(path) or os.path.getsize(path) <= 0:
        return False

    temp_path = f'{path}.h264.tmp.mp4'
    cmd = [
        FFMPEG_BINARY, '-y', '-i', path,
        '-c:v', 'libx264', '-preset', 'veryfast', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', '-an', temp_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
    except Exception:
        logger.exception('ffmpeg transcode failed to run for %s', path)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

    if result.returncode != 0 or not os.path.exists(temp_path) or os.path.getsize(temp_path) <= 0:
        logger.warning('ffmpeg transcode failed for %s: %s', path, result.stderr.decode(errors='ignore'))
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        return False

    # Retry replace on Windows to allow open file handles (e.g. OpenCV writer release or web reader) time to close
    for _ in range(5):
        try:
            os.replace(temp_path, path)
            return True
        except (PermissionError, OSError):
            time.sleep(0.3)

    try:
        import shutil
        shutil.copyfile(temp_path, path)
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        return True
    except Exception:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        return False


class _TranscodeQueue:
    """Background queue so ffmpeg does not block the camera capture thread."""

    def __init__(self):
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._worker, name='transcode-worker', daemon=True)
        self._started = False
        self._lock = threading.Lock()

    def start(self):
        with self._lock:
            if not self._started:
                self._thread.start()
                self._started = True

    def enqueue(self, path):
        if not path:
            return
        self.start()
        self._queue.put(path)

    def _worker(self):
        while True:
            path = self._queue.get()
            if path is None:
                self._queue.task_done()
                break
            try:
                transcode_to_h264(path)
            except Exception:
                logger.exception('Background transcode failed for %s', path)
            finally:
                self._queue.task_done()

    def stop(self, timeout=5):
        if not self._started:
            return
        try:
            self._queue.put(None, timeout=timeout)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)


_TRANSQUEUE = _TranscodeQueue()


def enqueue_transcode_to_h264(path):
    """Queue a file for background H.264 transcoding."""
    _TRANSQUEUE.enqueue(path)


def shutdown_transcode_queue(timeout=10):
    """Gracefully stop the background transcode worker."""
    try:
        _TRANSQUEUE.stop(timeout=timeout)
    except Exception:
        pass
