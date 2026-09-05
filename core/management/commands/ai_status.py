"""
Check and print AI hardware acceleration metrics (CUDA GPU vs CPU mode, PyTorch, OpenCV).

Usage:
    python manage.py ai_status
"""
from django.core.management.base import BaseCommand

from core.services.camera_service import get_ai_hardware_info


class Command(BaseCommand):
    help = 'Displays diagnostic info for AI Hardware Acceleration (PyTorch, CUDA GPU, OpenCV).'

    def handle(self, *args, **options):
        info = get_ai_hardware_info()

        self.stdout.write(self.style.MIGRATE_HEADING('=== VMS AI Hardware Acceleration Diagnostics ==='))
        self.stdout.write(f"Acceleration Mode  : {info['acceleration_mode']}")
        self.stdout.write(f"CUDA Available     : {'Yes' if info['cuda_available'] else 'No'}")
        self.stdout.write(f"Device Name        : {info['device_name']}")
        self.stdout.write(f"CUDA Driver Ver    : {info['cuda_version']}")
        self.stdout.write(f"PyTorch Version    : {info['torch_version']}")
        self.stdout.write(f"OpenCV Version     : {info['opencv_version']}")
        self.stdout.write(f"OpenCV CUDA Devs   : {info['opencv_cuda_devices']}")
        self.stdout.write('==================================================')
