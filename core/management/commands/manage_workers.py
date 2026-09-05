"""
Master Supervisor Command to manage background camera worker processes.

Usage:
    python manage.py manage_workers [--poll-interval SECONDS] [--dry-run]

Description:
    Auto-discovers enabled Camera models, spawns dedicated 'run_camera_worker'
    subprocesses for each, monitors process health, and auto-restarts failed workers.
"""
import os
import signal
import sys
import subprocess
import time

from django.core.management.base import BaseCommand
from core.models import Camera


class Command(BaseCommand):
    help = 'Master supervisor process to manage and monitor background camera workers.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--poll-interval',
            type=int,
            default=5,
            help='Seconds between process health checks (default: 5s).'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Inspect enabled cameras and simulate worker process spawning without running child processes.'
        )

    def handle(self, *args, **options):
        poll_interval = max(1, options['poll_interval'])
        dry_run = options['dry_run']

        self.stdout.write(self.style.MIGRATE_HEADING('=== VMS Background Camera Worker Supervisor ==='))
        if dry_run:
            self.stdout.write(self.style.WARNING('Running in DRY-RUN mode.'))

        workers = {}  # camera_id -> subprocess.Popen object

        def cleanup_and_exit(signum, frame):
            self.stdout.write(self.style.NOTICE('\nReceived shutdown signal. Stopping all camera workers...'))
            for cam_id, proc in list(workers.items()):
                if proc and proc.poll() is None:
                    self.stdout.write(f'Terminating worker process for Camera #{cam_id} (PID {proc.pid})...')
                    try:
                        proc.terminate()
                        proc.wait(timeout=5)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
            self.stdout.write(self.style.SUCCESS('All camera workers stopped. Supervisor exiting.'))
            sys.exit(0)

        try:
            signal.signal(signal.SIGINT, cleanup_and_exit)
            signal.signal(signal.SIGTERM, cleanup_and_exit)
        except (ValueError, AttributeError):
            pass

        python_executable = sys.executable
        manage_py_path = os.path.join(os.getcwd(), 'manage.py')

        try:
            while True:
                enabled_cameras = Camera.objects.filter(enabled=True)
                enabled_ids = set(enabled_cameras.values_list('id', flat=True))

                self.stdout.write(self.style.HTTP_INFO(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Monitoring {len(enabled_ids)} enabled camera(s)...'))

                # 1. Spawn missing workers for enabled cameras
                for camera in enabled_cameras:
                    cam_id = camera.id
                    proc = workers.get(cam_id)

                    # Check if existing process is dead
                    if proc is not None and proc.poll() is not None:
                        exit_code = proc.poll()
                        self.stdout.write(self.style.WARNING(f'Worker for Camera #{cam_id} ({camera.name}) exited with code {exit_code}. Restarting...'))
                        workers.pop(cam_id, None)
                        proc = None

                    if proc is None:
                        if dry_run:
                            self.stdout.write(self.style.SUCCESS(f'[DRY-RUN] Would launch: {python_executable} manage.py run_camera_worker {cam_id} ({camera.name})'))
                        else:
                            self.stdout.write(self.style.SUCCESS(f'Spawning worker process for Camera #{cam_id} ({camera.name})...'))
                            cmd = [python_executable, manage_py_path, 'run_camera_worker', str(cam_id)]
                            new_proc = subprocess.Popen(cmd)
                            workers[cam_id] = new_proc
                            self.stdout.write(f'Camera #{cam_id} worker started with PID {new_proc.pid}.')

                # 2. Terminate workers for cameras that were disabled
                for cam_id, proc in list(workers.items()):
                    if cam_id not in enabled_ids:
                        self.stdout.write(self.style.WARNING(f'Camera #{cam_id} is no longer enabled. Terminating worker process...'))
                        if not dry_run and proc and proc.poll() is None:
                            try:
                                proc.terminate()
                                proc.wait(timeout=5)
                            except Exception:
                                pass
                        workers.pop(cam_id, None)

                if dry_run:
                    self.stdout.write(self.style.SUCCESS('Dry run completed successfully.'))
                    break

                time.sleep(poll_interval)
        except KeyboardInterrupt:
            cleanup_and_exit(None, None)
