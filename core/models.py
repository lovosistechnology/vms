from config import settings
from django.db import models
from django.utils import timezone
from django.utils.text import slugify


class Camera(models.Model):
    SOURCE_TYPES = [
        ('built_in_webcam', 'Built-in webcam'),
        ('usb_webcam', 'USB webcam'),
        ('rtsp', 'RTSP camera'),
        ('http_mjpeg', 'HTTP/MJPEG camera'),
        ('local_video', 'Local video file'),
    ]
    STATUS_CHOICES = [
        ('offline', 'Offline'),
        ('online', 'Online'),
        ('error', 'Error'),
    ]

    name = models.CharField(max_length=120)
    source_type = models.CharField(max_length=20, choices=SOURCE_TYPES, default='built_in_webcam')
    source = models.CharField(
        max_length=500,
        blank=True,
        null=True,
        default=None,
        help_text='RTSP URL (rtsp://...), HTTP URL (http://...), webcam index (0, 1), or local video path.'
    )
    enabled = models.BooleanField(default=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='offline')
    last_error = models.TextField(blank=True, default='')

    @property
    def display_source(self):
        if self.source:
            return self.source
        if self.source_type == 'built_in_webcam':
            return '0 (Built-in)'
        if self.source_type == 'usb_webcam':
            return '1 (USB)'
        return 'Not configured'


    # Retention policy
    video_retention_days = models.IntegerField(default=30, help_text='Days to keep continuous recordings.')
    visit_profile_retention_days = models.IntegerField(default=90, help_text='Days to keep visit profile records.')

    # Alerting
    alert_on_offline = models.BooleanField(default=True, help_text='Show in-app alert when this camera goes offline.')

    # Worker management
    auto_start_worker = models.BooleanField(default=True, help_text='Whether this camera should start its worker automatically.')

    # Counting Zone Configuration (%)
    zone_enabled = models.BooleanField(default=True, help_text='Enable zone-restricted counting for this camera.')
    zone_x_min = models.IntegerField(default=5, help_text='Zone left boundary (%)')
    zone_y_min = models.IntegerField(default=5, help_text='Zone top boundary (%)')
    zone_x_max = models.IntegerField(default=95, help_text='Zone right boundary (%)')
    zone_y_max = models.IntegerField(default=95, help_text='Zone bottom boundary (%)')

    # Tripwire Line Crossing Configuration (%)
    tripwire_enabled = models.BooleanField(default=True, help_text='Enable line crossing tripwire for IN/OUT counting.')
    tripwire_x1 = models.IntegerField(default=10, help_text='Tripwire start X (%)')
    tripwire_y1 = models.IntegerField(default=50, help_text='Tripwire start Y (%)')
    tripwire_x2 = models.IntegerField(default=90, help_text='Tripwire end X (%)')
    tripwire_y2 = models.IntegerField(default=50, help_text='Tripwire end Y (%)')

    # Queue & Overcrowding Configuration
    queue_max_capacity = models.IntegerField(default=5, help_text='Max allowed persons in Zone before triggering overcrowding alarm.')
    queue_alert_enabled = models.BooleanField(default=True, help_text='Enable web audio chime and alert when zone is overcrowded.')
    heatmap_enabled = models.BooleanField(default=True, help_text='Accumulate real-time spatial traffic density heatmap.')

    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return self.name

    @property
    def slug(self):
        return slugify(self.name) or 'camera'


class Video(models.Model):
    camera = models.ForeignKey(Camera, related_name='videos', on_delete=models.CASCADE)
    file_path = models.CharField(max_length=500)
    created_at = models.DateTimeField(default=timezone.now)
    segment_start = models.DateTimeField(default=timezone.now)
    duration_seconds = models.IntegerField(default=0)
    frames_count = models.IntegerField(default=0)
    ready = models.BooleanField(default=False)

    def __str__(self):
        return f'{self.camera.name} - {self.file_path}'


class Visitor(models.Model):
    camera = models.ForeignKey(Camera, related_name='visitors', on_delete=models.CASCADE)
    video = models.ForeignKey(Video, related_name='visitors', on_delete=models.CASCADE, blank=True, null=True)
    track_id = models.IntegerField()
    entry_time = models.DateTimeField(default=timezone.now)
    counted_time = models.DateTimeField(default=timezone.now)
    dwell_time = models.IntegerField(default=0)

    class Meta:
        unique_together = [('camera', 'track_id')]

    def __str__(self):
        return f'{self.camera.name} track {self.track_id}'


class PersonProfile(models.Model):
    """A persistent person identity created from the first counted visit.

    Re-identification matches future visits to this profile using an appearance
    embedding so the same actual person does not create multiple profiles.
    """

    GENDER_CHOICES = [
        ('unknown', 'Unknown'),
        ('male', 'Male'),
        ('female', 'Female'),
    ]

    ATTIRE_CHOICES = [
        ('unknown', 'Unknown'),
        ('arabic_male', 'Arabic dress (Thobe/Kandura)'),
        ('arabic_female', 'Arabic dress (Abaya)'),
        ('modern_dress', 'Modern dress'),
    ]

    PERSON_TYPES = [
        ('regular', 'Regular Visitor'),
        ('vip', 'VIP Customer'),
        ('blacklist', 'Blacklisted Person'),
    ]

    camera = models.ForeignKey(Camera, related_name='person_profiles', on_delete=models.CASCADE)
    name = models.CharField(max_length=120, blank=True, default='', help_text='Custom display name or label for this person.')
    person_type = models.CharField(max_length=20, choices=PERSON_TYPES, default='regular', help_text='Classification label.')
    notes = models.TextField(blank=True, default='', help_text='Staff notes or security alerts.')
    first_seen = models.DateTimeField(default=timezone.now)
    last_seen = models.DateTimeField(default=timezone.now)
    visit_count = models.IntegerField(default=0)
    gender = models.CharField(max_length=10, choices=GENDER_CHOICES, default='unknown')
    attire = models.CharField(max_length=20, choices=ATTIRE_CHOICES, default='unknown', help_text='Most common attire observed for this person.')
    attire_attributes = models.JSONField(default=dict, blank=True, help_text='Clothing attributes from the latest classification.')
    embedding = models.JSONField(default=list, blank=True, help_text='Appearance embedding for re-identification.')

    class Meta:
        ordering = ['-last_seen']

    def __str__(self):
        display_name = self.name or f'Person #{self.pk}'
        return f'{self.camera.name} - {display_name}'


class VisitProfile(models.Model):
    """A daily visit profile for a camera.

    - Visit profiles are created after 60 seconds of continuous tracking.
    """

    GENDER_CHOICES = PersonProfile.GENDER_CHOICES
    ATTIRE_CHOICES = PersonProfile.ATTIRE_CHOICES

    camera = models.ForeignKey(Camera, related_name='visit_profiles', on_delete=models.CASCADE)
    person_profile = models.ForeignKey(PersonProfile, related_name='visits', on_delete=models.SET_NULL, blank=True, null=True)
    video = models.ForeignKey(Video, related_name='visit_profiles', on_delete=models.CASCADE, blank=True, null=True)
    sequence_number = models.IntegerField()
    track_id = models.IntegerField()
    date = models.DateField()
    entry_time = models.DateTimeField()
    counted_time = models.DateTimeField()
    dwell_time = models.IntegerField(default=0)
    required_seconds = models.IntegerField(default=60)
    gender = models.CharField(max_length=10, choices=GENDER_CHOICES, default='unknown')
    attire = models.CharField(max_length=20, choices=ATTIRE_CHOICES, default='unknown', help_text='Detected clothing category for this visit.')
    attire_attributes = models.JSONField(default=dict, blank=True, help_text='Clothing attributes such as colour, coverage, fit, formality.')
    tripwire_direction = models.CharField(max_length=10, default='none', help_text='Tripwire line crossing direction (in, out, or none).')
    snapshot_path = models.CharField(max_length=500, blank=True, default='')
    liveness_score = models.FloatField(default=0.0, help_text='Anti-spoofing confidence (higher is more likely real).')
    pose = models.CharField(max_length=30, blank=True, default='')

    class Meta:
        ordering = ['-counted_time']
        unique_together = [('camera', 'date', 'sequence_number')]

    @property
    def snapshot_url(self):
        if not self.snapshot_path:
            return ''
        clean_path = self.snapshot_path.replace('\\', '/').lstrip('/')
        if clean_path.startswith('media/'):
            clean_path = clean_path[6:]
        return f"{settings.MEDIA_URL.rstrip('/')}/{clean_path}"

    def __str__(self):
        return f'{self.camera.name} visit #{self.sequence_number} on {self.date}'



class SystemAlert(models.Model):
    """In-app alert shown to logged-in users."""

    camera = models.ForeignKey(Camera, related_name='alerts', on_delete=models.CASCADE, blank=True, null=True)
    title = models.CharField(max_length=200)
    message = models.TextField()
    created_at = models.DateTimeField(default=timezone.now)
    acknowledged = models.BooleanField(default=False)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return self.title
