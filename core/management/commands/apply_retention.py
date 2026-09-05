"""
Apply retention policies configured on each camera.

Usage:
    python manage.py apply_retention [--dry-run]

Deletes Video records (and their files) that are older than the camera's
video_retention_days setting.
Schedule this command via cron, systemd timer, or Celery beat.
"""
import os
from datetime import timedelta

from django.core.management.base import BaseCommand
from django.utils import timezone

from core.models import Camera, Video, VisitProfile


class Command(BaseCommand):
    help = 'Delete recordings and visit profiles older than each camera retention policy.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be deleted without actually deleting.',
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        total_videos = 0
        total_profiles = 0
        now = timezone.now()

        for camera in Camera.objects.all():
            video_days = max(1, camera.video_retention_days or 30)
            profile_days = max(1, camera.visit_profile_retention_days or 90)
            video_cutoff = now - timedelta(days=video_days)
            profile_cutoff = now - timedelta(days=profile_days)

            old_videos = Video.objects.filter(camera=camera, created_at__lt=video_cutoff)
            old_profiles = VisitProfile.objects.filter(camera=camera, date__lt=profile_cutoff)
            video_count = old_videos.count()
            profile_count = old_profiles.count()

            if dry_run:
                self.stdout.write(
                    f'Camera "{camera.name}": would delete {video_count} videos older than {video_cutoff} '
                    f'and {profile_count} visit profiles older than {profile_cutoff}'
                )
                total_videos += video_count
                total_profiles += profile_count
                continue

            for video in old_videos.iterator():
                if video.file_path and os.path.exists(video.file_path):
                    try:
                        os.remove(video.file_path)
                    except OSError:
                        pass
                video.delete()

            for profile in old_profiles.iterator():
                if profile.snapshot_path and os.path.exists(profile.snapshot_path):
                    try:
                        os.remove(profile.snapshot_path)
                    except OSError:
                        pass
                profile.delete()

            total_videos += video_count
            total_profiles += profile_count
            if video_count or profile_count:
                self.stdout.write(
                    f'Camera "{camera.name}": deleted {video_count} videos and {profile_count} visit profiles'
                )

        self.stdout.write(
            self.style.SUCCESS(f'Retention complete. Total videos: {total_videos}, total visit profiles: {total_profiles}.')
        )
