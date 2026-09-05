"""
Run the capture/analytics worker for a single camera in its own process.

Usage:
    python manage.py run_camera_worker <camera_id>

This command is intended for process isolation: run one instance per camera
so that a crash or CPU spike in one camera worker does not affect others.
In production, use systemd, supervisord, or Docker Compose to keep these
processes alive.
"""
import signal
import sys
import time

from django.core.management.base import BaseCommand, CommandError

from core.models import Camera
from core.services.camera_service import CameraService, shutdown_transcode_queue


class Command(BaseCommand):
    help = 'Run the capture worker for a single camera in a dedicated process.'

    def add_arguments(self, parser):
        parser.add_argument('camera_id', type=int, help='ID of the camera to run.')

    def handle(self, *args, **options):
        camera_id = options['camera_id']
        try:
            camera = Camera.objects.get(pk=camera_id)
        except Camera.DoesNotExist:
            raise CommandError(f'Camera with ID {camera_id} does not exist.')

        if not camera.enabled:
            self.stdout.write(self.style.WARNING(f'Camera {camera_id} is disabled. Exiting.'))
            return

        service = CameraService(camera_id)

        def on_signal(signum, frame):
            self.stdout.write(self.style.NOTICE(f'Received signal {signum}, stopping camera {camera_id}...'))
            service.stop()
            shutdown_transcode_queue(timeout=10)
            sys.exit(0)

        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)

        self.stdout.write(self.style.SUCCESS(f'Starting worker for camera {camera_id} ({camera.name})...'))
        if not service.start():
            self.stdout.write(self.style.ERROR(f'Failed to start camera {camera_id}: {service._last_error}'))
            # The service schedules its own restart; keep the process alive and wait.

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            service.stop()
            shutdown_transcode_queue(timeout=10)
