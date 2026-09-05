from django import forms

from .models import Camera


class CameraForm(forms.ModelForm):
    class Meta:
        model = Camera
        fields = [
            'name', 'source_type', 'source', 'enabled',
            'video_retention_days', 'visit_profile_retention_days', 'alert_on_offline',
            'auto_start_worker', 'zone_enabled', 'zone_x_min', 'zone_y_min',
            'zone_x_max', 'zone_y_max', 'tripwire_enabled', 'tripwire_x1', 'tripwire_y1',
            'tripwire_x2', 'tripwire_y2', 'queue_max_capacity',
        ]
        help_texts = {
            'video_retention_days': 'Days to keep continuous recordings.',
            'visit_profile_retention_days': 'Days to keep visit profile records.',
            'alert_on_offline': 'Show in-app alert when this camera goes offline.',
            'auto_start_worker': 'Whether this camera should start its worker automatically.',
            'zone_enabled': 'Enable zone-restricted counting for this camera.',
            'tripwire_enabled': 'Enable line crossing tripwire for IN/OUT counting.',
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Model defaults will be used when these fields are omitted.
        for field_name in (
            'video_retention_days', 'visit_profile_retention_days',
            'zone_x_min', 'zone_y_min', 'zone_x_max', 'zone_y_max',
            'tripwire_x1', 'tripwire_y1', 'tripwire_x2', 'tripwire_y2',
            'queue_max_capacity'
        ):
            if field_name in self.fields:
                self.fields[field_name].required = False

    def clean(self):
        cleaned_data = super().clean()
        source_type = cleaned_data.get('source_type')
        source = (cleaned_data.get('source') or '').strip()

        if source_type in ('rtsp', 'http_mjpeg', 'local_video') and not source:
            type_label = dict(Camera.SOURCE_TYPES).get(source_type, source_type)
            self.add_error('source', f'A valid stream URL or file path is required for {type_label}.')
        elif source_type == 'usb_webcam' and not source:
            cleaned_data['source'] = '1'
        elif source_type == 'built_in_webcam' and not source:
            cleaned_data['source'] = '0'
        else:
            cleaned_data['source'] = source

        return cleaned_data

    def save(self, commit=True):
        instance = super().save(commit=False)
        if instance.source_type == 'built_in_webcam' and not instance.source:
            instance.source = '0'
        if commit:
            instance.save()
        return instance


