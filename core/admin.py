from django.contrib import admin

from .models import Camera, PersonProfile, SystemAlert, Video, Visitor, VisitProfile


@admin.register(Camera)
class CameraAdmin(admin.ModelAdmin):
    list_display = ('name', 'source_type', 'source', 'status', 'enabled', 'created_at')
    list_filter = ('source_type', 'status', 'enabled', 'created_at')
    search_fields = ('name', 'source')
    readonly_fields = ('status', 'last_error', 'created_at', 'updated_at')
    fieldsets = (
        ('Camera Info', {'fields': ('name', 'source_type', 'source', 'enabled', 'auto_start_worker')}),
        ('Counting Zone Setup', {'fields': ('zone_enabled', 'zone_x_min', 'zone_y_min', 'zone_x_max', 'zone_y_max', 'queue_max_capacity', 'queue_alert_enabled')}),
        ('Tripwire Setup', {'fields': ('tripwire_enabled', 'tripwire_x1', 'tripwire_y1', 'tripwire_x2', 'tripwire_y2')}),
        ('Retention', {'fields': ('video_retention_days', 'visit_profile_retention_days')}),
        ('Alerts', {'fields': ('alert_on_offline',)}),
        ('Status', {'fields': ('status', 'last_error'), 'classes': ('collapse',)}),
        ('Timestamps', {'fields': ('created_at', 'updated_at'), 'classes': ('collapse',)}),
    )

    def save_model(self, request, obj, form, change):
        if obj.source:
            obj.source = obj.source.strip()
        if obj.source_type == 'built_in_webcam' and not obj.source:
            obj.source = '0'
        elif obj.source_type == 'usb_webcam' and not obj.source:
            obj.source = '1'
        super().save_model(request, obj, form, change)



@admin.register(Video)
class VideoAdmin(admin.ModelAdmin):
    list_display = ('camera', 'file_path', 'created_at', 'duration_seconds', 'frames_count')
    list_filter = ('camera', 'created_at')
    search_fields = ('camera__name', 'file_path')
    readonly_fields = ('created_at',)


@admin.register(Visitor)
class VisitorAdmin(admin.ModelAdmin):
    list_display = ('camera', 'track_id', 'entry_time', 'counted_time', 'dwell_time')
    list_filter = ('camera', 'counted_time', 'entry_time')
    search_fields = ('camera__name',)
    readonly_fields = ('entry_time', 'counted_time')


@admin.register(VisitProfile)
class VisitProfileAdmin(admin.ModelAdmin):
    list_display = ('camera', 'sequence_number', 'track_id', 'gender', 'attire', 'pose', 'counted_time', 'dwell_time')
    list_filter = ('camera', 'gender', 'attire', 'pose', 'date')
    search_fields = ('camera__name',)
    readonly_fields = ('entry_time', 'counted_time', 'snapshot_path', 'liveness_score', 'attire_attributes')


@admin.register(PersonProfile)
class PersonProfileAdmin(admin.ModelAdmin):
    list_display = ('camera', 'visit_count', 'gender', 'attire', 'first_seen', 'last_seen')
    list_filter = ('camera', 'gender', 'attire')
    search_fields = ('camera__name',)
    readonly_fields = ('attire_attributes',)


@admin.register(SystemAlert)
class SystemAlertAdmin(admin.ModelAdmin):
    list_display = ('title', 'camera', 'created_at', 'acknowledged')
    list_filter = ('acknowledged', 'created_at')
    search_fields = ('title', 'message')

