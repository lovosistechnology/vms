import os
import sys
import threading
import time
from django.apps import AppConfig


class CoreConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'core'

    def ready(self):
        """Auto-start camera background workers when running 'python manage.py runserver'."""
        is_runserver = any('runserver' in arg for arg in sys.argv)
        is_active_process = os.environ.get('RUN_MAIN') == 'true' or '--noreload' in sys.argv

        if is_runserver and is_active_process:
            def _auto_start_workers():
                time.sleep(1.5)  # Wait for Django server startup to complete
                try:
                    from core.models import Camera
                    from core.services.camera_service import CameraService
                    enabled_cameras = Camera.objects.filter(enabled=True, auto_start_worker=True)
                    if enabled_cameras.exists():
                        print(f"\n[VMS Engine] Auto-starting {enabled_cameras.count()} camera worker(s) in-process...")
                        for camera in enabled_cameras:
                            service = CameraService(camera.id)
                            service.start()
                except Exception as exc:
                    print(f"[VMS Engine] Auto-worker startup notice: {exc}")

            worker_thread = threading.Thread(
                target=_auto_start_workers,
                name='vms-auto-worker-daemon',
                daemon=True,
            )
            worker_thread.start()
