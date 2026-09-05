from django.contrib.auth.views import LogoutView
from django.urls import path

from . import views

app_name = 'core'

urlpatterns = [
    path('', views.login_view, name='login'),
    path('logout/', LogoutView.as_view(next_page='core:login'), name='logout'),

    path('dashboard/', views.dashboard, name='dashboard'),
    path('live/', views.all_cameras_live, name='all_cameras_live'),
    path('analytics/', views.analytics, name='analytics'),
    path('analytics/pdf/', views.export_analytics_pdf, name='export_analytics_pdf'),

    path('cameras/', views.cameras, name='cameras'),
    path('camera/add/', views.add_camera, name='add_camera'),
    path('camera/<int:pk>/edit/', views.edit_camera, name='edit_camera'),
    path('camera/<int:pk>/delete/', views.delete_camera, name='delete_camera'),
    path('camera/<int:pk>/live/', views.live_camera, name='live_camera'),
    path('camera/<int:pk>/state/', views.camera_state, name='camera_state'),
    path('camera/<int:pk>/stream/', views.camera_stream, name='camera_stream'),
    path('camera/<int:pk>/recordings/', views.recordings, name='recordings'),
    path('camera/<int:pk>/profiles/export/', views.export_visit_profiles, name='export_visit_profiles'),
    path('camera/<int:pk>/update_geometry/', views.update_camera_geometry, name='update_camera_geometry'),

    path('search/', views.search_by_photo_view, name='search_by_photo'),
    path('person/<int:pk>/edit/', views.edit_person_profile, name='edit_person_profile'),

    path('alerts/acknowledge/<int:pk>/', views.acknowledge_alert, name='acknowledge_alert'),
    path('recordings/<int:pk>/play/', views.play_recording, name='play_recording'),
    path('recordings/<int:pk>/stream/', views.stream_recording, name='stream_recording'),
]
