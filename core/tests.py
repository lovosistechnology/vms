import os
import tempfile
import time
from datetime import timedelta
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from django.contrib.auth import get_user_model
from django.test import RequestFactory, TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from . import views
from .models import Camera, PersonProfile, Video, Visitor, VisitProfile
from .services.clothing_service import ATTIRE_ARABIC_FEMALE, ATTIRE_ARABIC_MALE, ATTIRE_MODERN, get_clothing_service


@override_settings(MEDIA_ROOT=tempfile.gettempdir())
class PlaybackViewsTests(TestCase):
    def test_analytics_data_aggregation_is_query_efficient(self):
        start_date = timezone.now().date() - timedelta(days=6)
        end_date = timezone.now().date()
        camera = Camera.objects.create(name='Analytics Camera', source_type='built_in_webcam', source='0', enabled=True)

        for idx in range(3):
            VisitProfile.objects.create(
                camera=camera,
                person_profile=None,
                video=None,
                sequence_number=idx + 1,
                track_id=idx + 1,
                date=start_date + timedelta(days=idx % 2),
                entry_time=timezone.now(),
                counted_time=timezone.now(),
                dwell_time=60 + idx,
                gender='male' if idx % 2 == 0 else 'female',
                attire='arabic_male' if idx == 0 else 'modern_dress',
            )

        PersonProfile.objects.create(camera=camera, first_seen=timezone.now(), last_seen=timezone.now())

        with self.assertNumQueries(8):
            data = views._get_analytics_data(start_date, end_date)

        self.assertEqual(data['total_visits'], 3)
        self.assertEqual(len(data['visits_by_date']), 7)

    @staticmethod
    def _create_test_video(path):
        """Create a small, valid MP4 so OpenCV can read it during streaming tests."""
        writer = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            10.0,
            (64, 48),
        )
        if not writer.isOpened():
            raise RuntimeError('Unable to create test MP4 fixture')
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
        writer.release()

    def setUp(self):
        self.user = get_user_model().objects.create_user(username='tester', password='secret123')
        self.factory = RequestFactory()
        self.client.force_login(self.user)
        self.camera = Camera.objects.create(name='Test Camera', source_type='built_in_webcam', source='0', enabled=True)
        self.video_path = os.path.join(tempfile.gettempdir(), 'sample-recording.mp4')
        self._create_test_video(self.video_path)
        self.video = Video.objects.create(camera=self.camera, file_path=self.video_path, ready=True)

    def test_recording_playback_page_renders(self):
        response = self.client.get(reverse('core:play_recording', args=[self.video.pk]))
        self.assertEqual(response.status_code, 200)
        self.assertIn('Recording Playback', response.content.decode('utf-8'))

    def test_recording_stream_serves_file(self):
        response = self.client.get(reverse('core:stream_recording', args=[self.video.pk]))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'video/mp4')

    def test_camera_rotation_starts_a_new_segment_without_error(self):
        from .services.camera_service import CameraService

        service = CameraService(self.camera.pk)
        service._latest_frame = np.zeros((48, 64, 3), dtype=np.uint8)
        service._segment_start = time.time() - 121
        service._writer = MagicMock()
        service._recording = True
        service._active_video = None

        with patch('cv2.VideoWriter_fourcc', return_value='mp4v'), patch('cv2.VideoWriter') as writer_cls:
            writer = MagicMock()
            writer.isOpened.return_value = True
            writer_cls.return_value = writer
            service._rotate_recording(self.camera)

        self.assertIs(service._writer, writer)
        self.assertTrue(service._recording)
        self.assertTrue(service._segment_path.endswith('.mp4'))

    def test_recording_segment_rotates_after_short_interval(self):
        from .services.camera_service import CameraService

        service = CameraService(self.camera.pk)
        service._latest_frame = np.zeros((48, 64, 3), dtype=np.uint8)
        service._segment_start = time.time() - 31
        service._writer = MagicMock()
        service._recording = True
        service._active_video = Video.objects.create(
            camera=self.camera,
            file_path='/tmp/segment.mp4',
            ready=False,
            frames_count=1,
            duration_seconds=1,
        )

        with patch('cv2.VideoWriter_fourcc', return_value='mp4v'), patch('cv2.VideoWriter') as writer_cls:
            writer = MagicMock()
            writer.isOpened.return_value = True
            writer_cls.return_value = writer
            service._write_frame(service._latest_frame, self.camera)

        self.assertTrue(service._recording)
        self.assertTrue(service._segment_path.endswith('.mp4'))
        self.assertIsNotNone(service._active_video)

    def test_visitor_is_counted_after_60_seconds_and_linked_to_active_video(self):
        from datetime import timedelta

        from django.utils import timezone

        from .services.camera_service import CameraService

        service = CameraService(self.camera.pk)
        video = Video.objects.create(camera=self.camera, file_path='/tmp/segment.mp4', ready=False)
        service._active_video = video

        now = timezone.now()
        service._trackers[1] = {
            'first_seen': now - timedelta(seconds=65),
            'last_seen': now,
            'counted': False,
            'visitor_id': None,
            'highlight_until': None,
            'required_seconds': 60,
        }

        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        service._update_track_state(self.camera, 1, 10, 10, 50, 50, dummy_frame)

        self.assertTrue(service._trackers[1]['counted'])
        self.assertIsNotNone(service._trackers[1]['visitor_id'])
        visitor = Visitor.objects.get(pk=service._trackers[1]['visitor_id'])
        self.assertEqual(visitor.camera, self.camera)
        self.assertEqual(visitor.video, video)
        self.assertGreaterEqual(visitor.dwell_time, 60)

    def test_camera_state_endpoint_returns_live_status_payload(self):
        response = self.client.get(reverse('core:camera_state', args=[self.camera.pk]))

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('recording', data)
        self.assertIn('motion_detected', data)
        self.assertIn('recording_duration_display', data)

    @patch('core.views.get_service')
    def test_all_cameras_live_renders_enabled_cameras_from_one_url(self, get_service):
        service = MagicMock()
        get_service.return_value = service
        second_camera = Camera.objects.create(name='Second Camera', source_type='rtsp', source='rtsp://example/live')
        Camera.objects.create(name='Disabled Camera', enabled=False)

        response = self.client.get(reverse('core:all_cameras_live'))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Test Camera')
        self.assertContains(response, 'Second Camera')
        self.assertNotContains(response, 'Disabled Camera')
        self.assertContains(response, reverse('core:camera_stream', args=[self.camera.pk]))
        self.assertContains(response, reverse('core:camera_stream', args=[second_camera.pk]))
        self.assertEqual(get_service.call_count, 2)
        self.assertEqual(service.start.call_count, 2)

    def test_cameras_page_renders_csrf_protection_markup(self):
        request = self.factory.get(reverse('core:cameras'))
        request.user = self.user

        response = views.cameras(request)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'csrfmiddlewaretoken')

    def test_add_camera_view_creates_camera_and_returns_to_listing(self):
        request = self.factory.post(
            reverse('core:add_camera'),
            {'name': 'Popup Camera', 'source_type': 'built_in_webcam', 'enabled': 'on'}
        )
        request.user = self.user

        response = views.add_camera(request)

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse('core:cameras'))
        self.assertTrue(Camera.objects.filter(name='Popup Camera').exists())

    @patch('core.views.get_service')
    def test_add_camera_expands_rtsp_unicast_base_into_channels(self, get_service):
        request = self.factory.post(
            reverse('core:add_camera'),
            {
                'name': 'NVR',
                'source_type': 'rtsp',
                'source': 'rtsp://admin:admin%40123@100.118.229.80:554/unicast',
                'enabled': 'on',
            },
        )
        request.user = self.user
        get_service.return_value.start.return_value = True

        response = views.add_camera(request)

        cameras = Camera.objects.filter(name__startswith='NVR - Channel ').order_by('name')
        self.assertEqual(response.status_code, 302)
        self.assertEqual(cameras.count(), 4)
        self.assertEqual(
            cameras.filter(source='rtsp://admin:admin%40123@100.118.229.80:554/unicast/c1/s1/live').count(),
            1,
        )
        self.assertEqual(get_service.call_count, 4)

    def test_visitor_profile_stores_attire_after_counting(self):
        """A counted visitor should have an attire value stored on VisitProfile."""
        from datetime import timedelta

        from django.utils import timezone

        from .services.camera_service import CameraService

        service = CameraService(self.camera.pk)
        video = Video.objects.create(camera=self.camera, file_path='/tmp/segment.mp4', ready=False)
        service._active_video = video

        now = timezone.now()
        service._trackers[1] = {
            'first_seen': now - timedelta(seconds=65),
            'last_seen': now,
            'counted': False,
            'visitor_id': None,
            'highlight_until': None,
            'required_seconds': 60,
            'gender_votes': {'male': 0, 'female': 0, 'unknown': 0},
        }

        # White frame should be classified as Arabic male attire.
        white_frame = np.full((200, 100, 3), 255, dtype=np.uint8)
        service._update_track_state(self.camera, 1, 10, 10, 90, 190, white_frame)

        self.assertTrue(service._trackers[1]['counted'])
        visit = VisitProfile.objects.filter(camera=self.camera, track_id=1).first()
        self.assertIsNotNone(visit)
        self.assertIn(visit.attire, [ATTIRE_ARABIC_MALE, ATTIRE_MODERN, ATTIRE_ARABIC_FEMALE])
        self.assertIsNotNone(visit.attire_attributes)

    def test_clothing_service_detects_white_thobe(self):
        """A mostly white long garment should be classified as Arabic male dress."""
        service = get_clothing_service()
        white_frame = np.full((200, 100, 3), 255, dtype=np.uint8)
        result = service.classify(white_frame, 10, 10, 90, 190, gender='male')
        self.assertEqual(result['category'], ATTIRE_ARABIC_MALE)
        self.assertEqual(result['attributes']['dominant_colour'], 'white')

    def test_clothing_service_detects_black_abaya(self):
        """A mostly black long garment should be classified as Arabic female dress."""
        service = get_clothing_service()
        black_frame = np.full((200, 100, 3), 15, dtype=np.uint8)
        result = service.classify(black_frame, 10, 10, 90, 190, gender='female')
        self.assertEqual(result['category'], ATTIRE_ARABIC_FEMALE)
        self.assertEqual(result['attributes']['dominant_colour'], 'black')

    def test_return_visitor_is_counted_after_60_seconds(self):
        """Return visits should require 60 seconds (same as first visit)."""
        from .services.camera_service import CameraService, _VISIT_SECONDS

        service = CameraService(self.camera.pk)
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Create an initial visit profile today
        VisitProfile.objects.create(
            camera=self.camera,
            sequence_number=1,
            track_id=10,
            date=timezone.now().date(),
            entry_time=timezone.now(),
            counted_time=timezone.now(),
            dwell_time=60,
            required_seconds=_VISIT_SECONDS,
        )

        # Initializing a new track should assign _VISIT_SECONDS (60)
        service._trackers.clear()
        service._update_track_state(self.camera, 2, 10, 10, 50, 50, dummy_frame)
        self.assertEqual(service._trackers[2]['required_seconds'], 60)

    def test_zone_entry_exit_reentry_counting_rules(self):
        """Test zone entry 60s count, exit reset, and re-entry 60s count."""
        from datetime import timedelta

        from .services.camera_service import CameraService

        # Camera with zone enabled (10% to 90%)
        self.camera.zone_enabled = True
        self.camera.zone_x_min = 10
        self.camera.zone_y_min = 10
        self.camera.zone_x_max = 90
        self.camera.zone_y_max = 90
        self.camera.save()

        service = CameraService(self.camera.pk)
        service._trackers.clear()
        frame = np.zeros((200, 200, 3), dtype=np.uint8)

        # 1. Person outside zone (center at x=5, y=5)
        service._update_track_state(self.camera, 1, 0, 0, 10, 10, frame)
        self.assertFalse(service._trackers[1]['in_zone'])

        # Fast forward time outside zone -> should not count
        service._trackers[1]['first_seen'] = timezone.now() - timedelta(seconds=70)
        service._update_track_state(self.camera, 1, 0, 0, 10, 10, frame)
        self.assertFalse(service._trackers[1]['zone_counted'])

        # 2. Enter zone (center at x=100, y=100) -> Starts zone timer
        service._update_track_state(self.camera, 1, 80, 80, 120, 120, frame)
        self.assertTrue(service._trackers[1]['in_zone'])
        self.assertIsNotNone(service._trackers[1]['zone_entry_time'])

        # Simulate 65 seconds inside zone -> Should count as visit #1!
        service._trackers[1]['zone_entry_time'] = timezone.now() - timedelta(seconds=65)
        service._update_track_state(self.camera, 1, 80, 80, 120, 120, frame)
        self.assertTrue(service._trackers[1]['zone_counted'])
        self.assertEqual(service._trackers[1]['person_visit_count'], 1)
        self.assertEqual(VisitProfile.objects.filter(camera=self.camera).count(), 1)

        # 3. Person moves OUT of zone -> Reset zone state after hysteresis threshold (5 frames)
        for _ in range(5):
            service._update_track_state(self.camera, 1, 0, 0, 10, 10, frame)
        self.assertFalse(service._trackers[1]['in_zone'])
        self.assertFalse(service._trackers[1]['zone_counted'])
        self.assertIsNone(service._trackers[1]['zone_entry_time'])

        # 4. Person COMES BACK into zone -> New zone entry session
        service._update_track_state(self.camera, 1, 80, 80, 120, 120, frame)
        self.assertTrue(service._trackers[1]['in_zone'])
        self.assertFalse(service._trackers[1]['zone_counted'])

        # Simulate 65 seconds inside zone after re-entry -> Should count visit #2!
        service._trackers[1]['zone_entry_time'] = timezone.now() - timedelta(seconds=65)
        service._update_track_state(self.camera, 1, 80, 80, 120, 120, frame)
        self.assertTrue(service._trackers[1]['zone_counted'])
        self.assertEqual(service._trackers[1]['person_visit_count'], 2)
        self.assertEqual(VisitProfile.objects.filter(camera=self.camera).count(), 2)

    def test_photo_search_service(self):
        """Photo search service should extract embedding and rank profile matches."""
        from .services.search_service import get_search_service
        person = PersonProfile.objects.create(
            camera=self.camera,
            first_seen=timezone.now(),
            last_seen=timezone.now(),
            embedding=[0.1] * 48,
        )
        service = get_search_service()
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        ok, buf = cv2.imencode('.jpg', dummy_img)
        self.assertTrue(ok)
        matches = service.search_profiles_by_image(buf.tobytes(), min_score=0.0)
        self.assertTrue(len(matches) >= 1)

    def test_update_camera_geometry_endpoint(self):
        """API endpoint should update zone & tripwire coordinates."""
        response = self.client.post(
            reverse('core:update_camera_geometry', args=[self.camera.pk]),
            data={'zone_enabled': True, 'zone_x_min': 20, 'tripwire_enabled': True, 'tripwire_y1': 60},
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 200)
        self.camera.refresh_from_db()
        self.assertTrue(self.camera.zone_enabled)
    def test_add_rtsp_camera_form_validation_and_creation(self):
        """Adding an RTSP camera requires a valid source stream URL."""
        # 1. Invalid submit without source URL should fail validation
        response = self.client.post(reverse('core:add_camera'), data={'name': 'IP Cam 1', 'source_type': 'rtsp', 'source': ''})
        self.assertEqual(response.status_code, 200)
        self.assertIn('A valid stream URL or file path is required', response.content.decode('utf-8'))

        # 2. Valid submit with RTSP URL should create camera record
        with patch('core.views.get_service') as mock_get_service:
            mock_service = MagicMock()
            mock_get_service.return_value = mock_service
            response = self.client.post(reverse('core:add_camera'), data={'name': 'Front Gate IP Cam', 'source_type': 'rtsp', 'source': 'rtsp://admin:12345@192.168.1.100:554/stream1', 'enabled': True})
            self.assertEqual(response.status_code, 302)

        created = Camera.objects.get(name='Front Gate IP Cam')
        self.assertEqual(created.source_type, 'rtsp')
        self.assertEqual(created.source, 'rtsp://admin:12345@192.168.1.100:554/stream1')

    def test_delete_camera_endpoint(self):
        """Deleting a camera should remove it from the database."""
        cam = Camera.objects.create(name='Cam to Delete', source_type='rtsp', source='rtsp://192.168.1.200/live')
        response = self.client.post(reverse('core:delete_camera', args=[cam.pk]))
        self.assertEqual(response.status_code, 302)
        self.assertFalse(Camera.objects.filter(pk=cam.pk).exists())




