import csv
import os
import threading
import time
from datetime import datetime, timedelta
from io import BytesIO
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np
from django.conf import settings
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from django.db.models import Avg, Count, Q, Sum
from django.db.models.functions import ExtractHour
from django.http import FileResponse, Http404, HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.utils.html import escape
from django.views.decorators.http import require_GET, require_POST
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import inch
from reportlab.platypus import Image as ReportImage, PageBreak, SimpleDocTemplate, Spacer, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart, HorizontalBarChart
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics import renderPM

from .forms import CameraForm
from .models import Camera, PersonProfile, SystemAlert, Video, VisitProfile, Visitor
from .services.camera_service import CameraService, get_ai_hardware_info
from .services.video_utils import transcode_to_h264


SERVICE_CACHE = {}
DEFAULT_RTSP_CHANNEL_COUNT = 4


def _rtsp_channel_urls(source):
    """Return channel URLs when the user entered an NVR /unicast base URL."""
    parsed = urlsplit((source or '').strip())
    if parsed.scheme.lower() not in {'rtsp', 'rtsps'} or parsed.path.rstrip('/').lower() != '/unicast':
        return []

    channel_count = getattr(settings, 'VMS_RTSP_CHANNEL_COUNT', DEFAULT_RTSP_CHANNEL_COUNT)
    try:
        channel_count = max(1, int(channel_count))
    except (TypeError, ValueError):
        channel_count = DEFAULT_RTSP_CHANNEL_COUNT

    base_path = parsed.path.rstrip('/')
    return [
        urlunsplit(parsed._replace(path=f'{base_path}/c{channel}/s1/live'))
        for channel in range(1, channel_count + 1)
    ]


def _render_simple_page(title, body):
    return HttpResponse(
        f"<html><head><title>{escape(title)}</title></head><body>{body}</body></html>",
        content_type='text/html; charset=utf-8',
    )


def _create_fallback_mp4(path, *, label='video'):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(path, fourcc, 10.0, (640, 360))
        if not writer.isOpened():
            return False
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(frame, label, (80, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)
        writer.write(frame)
        writer.release()
        if not os.path.exists(path):
            return False
        transcode_to_h264(path)
        return True
    except Exception:
        return False


def _is_playable_mp4(path):
    if not path or not os.path.exists(path) or os.path.getsize(path) <= 0:
        return False
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return False
    ok, frame = cap.read()
    cap.release()
    return bool(ok and frame is not None)


def _stream_file_range(path, start, end, chunk_size=8192):
    with open(path, 'rb') as handle:
        handle.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            chunk = handle.read(min(chunk_size, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def _video_file_response(request, path, download_name):
    file_size = os.path.getsize(path)
    range_header = request.META.get('HTTP_RANGE', '').strip()

    if range_header.startswith('bytes='):
        range_value = range_header.replace('bytes=', '', 1).split(',', 1)[0].strip()
        if '-' in range_value:
            start_str, end_str = range_value.split('-', 1)
            try:
                if start_str == '':
                    length = int(end_str)
                    start = max(0, file_size - length)
                    end = file_size - 1
                else:
                    start = int(start_str)
                    end = int(end_str) if end_str else file_size - 1
                start = max(0, min(start, file_size - 1))
                end = max(start, min(end, file_size - 1))
                length = end - start + 1

                response = StreamingHttpResponse(
                    _stream_file_range(path, start, end),
                    status=206,
                    content_type='video/mp4',
                )
                response['Content-Range'] = f'bytes {start}-{end}/{file_size}'
                response['Content-Length'] = str(length)
                response['Accept-Ranges'] = 'bytes'
                response['Content-Disposition'] = f'inline; filename="{download_name}"'
                response['Cache-Control'] = 'no-store'
                response['X-Content-Type-Options'] = 'nosniff'
                return response
            except (TypeError, ValueError):
                pass

    response = FileResponse(open(path, 'rb'), content_type='video/mp4')
    response['Content-Length'] = str(file_size)
    response['Content-Disposition'] = f'inline; filename="{download_name}"'
    response['Accept-Ranges'] = 'bytes'
    response['Cache-Control'] = 'no-store'
    response['X-Content-Type-Options'] = 'nosniff'
    return response


def login_view(request):
    if request.user.is_authenticated:
        return redirect('core:dashboard')
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '').strip()
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('core:dashboard')
        return render(request, 'core/login.html', {'title': 'Login', 'error': 'Invalid username or password.'})
    return render(request, 'core/login.html', {'title': 'Login'})


def _get_camera_list_data():
    today = timezone.now().date()
    cameras_qs = Camera.objects.all()
    camera_list = []
    for camera in cameras_qs:
        today_visits = VisitProfile.objects.filter(camera=camera, date=today)
        attire_counts = {}
        for profile in today_visits:
            attire_counts[profile.attire] = attire_counts.get(profile.attire, 0) + 1
        camera_list.append({
            'camera': camera,
            'today_visits': today_visits.count(),
            'attire_counts': attire_counts,
        })
    return camera_list


@login_required
def cameras(request):
    camera_list = _get_camera_list_data()
    form = CameraForm()

    if request.method == 'POST':
        camera_id = request.POST.get('camera_id')
        if camera_id:
            camera = Camera.objects.get(pk=camera_id)
            if not camera.enabled:
                camera.enabled = True
                camera.save(update_fields=['enabled'])
            from django.core.cache import cache
            has_live = bool(cache.get(f'vms:live_frame:{camera.pk}'))
            service = get_service(camera.pk)
            if not has_live and not (service._thread and service._thread.is_alive()):
                service.start()
            return redirect('core:live_camera', pk=camera.pk)

    return render(request, 'core/cameras.html', {'cameras': camera_list, 'form': form})



@login_required
def dashboard(request):
    """Multi-camera dashboard showing status and today's visitor counts."""
    today = timezone.now().date()
    cameras = Camera.objects.all()
    dashboard_data = []
    total_visits = 0
    total_person_profiles = 0

    for camera in cameras:
        visit_count = VisitProfile.objects.filter(camera=camera, date=today).count()
        person_count = PersonProfile.objects.filter(camera=camera).count()
        total_visits += visit_count
        total_person_profiles += person_count

        # Aggregate today's attire counts for this camera.
        today_profiles = VisitProfile.objects.filter(camera=camera, date=today)
        attire_counts = {}
        for profile in today_profiles:
            attire_counts[profile.attire] = attire_counts.get(profile.attire, 0) + 1

        dashboard_data.append({
            'camera': camera,
            'today_visits': visit_count,
            'person_profiles': person_count,
            'attire_counts': attire_counts,
        })

    # Global attire aggregation for today.
    all_today_profiles = VisitProfile.objects.filter(date=today)
    total_attire_counts = {}
    for profile in all_today_profiles:
        total_attire_counts[profile.attire] = total_attire_counts.get(profile.attire, 0) + 1

    unacknowledged_alerts = SystemAlert.objects.filter(acknowledged=False)[:10]

    return render(request, 'core/dashboard.html', {
        'cameras': dashboard_data,
        'total_cameras': cameras.count(),
        'total_visits': total_visits,
        'total_person_profiles': total_person_profiles,
        'total_attire_counts': total_attire_counts,
        'alerts': unacknowledged_alerts,
        'ai_hardware': get_ai_hardware_info(),
    })


@login_required
def all_cameras_live(request):
    """Show every enabled camera live from one application URL."""
    cameras = list(Camera.objects.filter(enabled=True).order_by('name'))
    from django.core.cache import cache
    for camera in cameras:
        has_live = bool(cache.get(f'vms:live_frame:{camera.pk}'))
        service = get_service(camera.pk)
        if not has_live and not (service._thread and service._thread.is_alive()):
            service.start()
    return render(request, 'core/live.html', {'camera': None, 'cameras': cameras})


@login_required
def acknowledge_alert(request, pk):
    alert = get_object_or_404(SystemAlert, pk=pk)
    alert.acknowledged = True
    alert.save(update_fields=['acknowledged'])
    return redirect(request.META.get('HTTP_REFERER') or 'core:dashboard')


_GENDER_LABELS = {'male': 'Male', 'female': 'Female', 'unknown': 'Unknown'}
_ATTIRE_LABELS = {
    'arabic_male': 'Arabic Male',
    'arabic_female': 'Arabic Female',
    'modern_dress': 'Modern Dress',
    'unknown': 'Unknown',
}


def _parse_analytics_date_range(request):
    """Parse date range for analytics views."""
    today = timezone.now().date()
    range_key = request.GET.get('range', '7days')
    if range_key == 'custom':
        try:
            start = datetime.strptime(request.GET.get('start_date', ''), '%Y-%m-%d').date()
            end = datetime.strptime(request.GET.get('end_date', ''), '%Y-%m-%d').date()
        except (ValueError, TypeError):
            start = today - timedelta(days=6)
            end = today
        return start, end, f'{start} to {end}'
    days = {'7days': 7, '30days': 30, '90days': 90}.get(range_key, 7)
    start = today - timedelta(days=days - 1)
    end = today
    return start, end, f'Last {days} days'


def _get_analytics_data(start_date, end_date):
    """Aggregate visit analytics for the given date range."""
    cache_key = f'analytics:{start_date.isoformat()}:{end_date.isoformat()}'
    cached = SERVICE_CACHE.get(cache_key)
    if cached and time.time() - cached['timestamp'] < 60:
        return cached['data']

    profiles_qs = VisitProfile.objects.filter(date__gte=start_date, date__lte=end_date).only(
        'date', 'camera', 'counted_time', 'dwell_time', 'gender', 'attire'
    )

    stats = profiles_qs.aggregate(
        total_visits=Count('id'),
        avg_dwell=Avg('dwell_time'),
        total_dwell=Sum('dwell_time'),
    )
    person_stats = PersonProfile.objects.filter(
        first_seen__date__gte=start_date, first_seen__date__lte=end_date
    ).aggregate(new_persons=Count('id'))
    visits_by_date_raw = {
        item['date']: item['count']
        for item in profiles_qs.values('date').annotate(count=Count('id')).order_by('date')
    }
    date_range = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    visits_by_date = [
        {'date': d.strftime('%Y-%m-%d'), 'count': visits_by_date_raw.get(d, 0)}
        for d in date_range
    ]

    visits_by_camera = list(
        profiles_qs.values('camera__name')
        .annotate(count=Count('id'))
        .order_by('-count')
    )

    hour_raw = {
        item['hour']: item['count']
        for item in profiles_qs.annotate(hour=ExtractHour('counted_time'))
        .values('hour')
        .annotate(count=Count('id'))
        .order_by('hour')
        if item['hour'] is not None
    }
    visits_by_hour = [{'hour': h, 'count': hour_raw.get(h, 0)} for h in range(24)]

    gender_counts = list(profiles_qs.values('gender').annotate(count=Count('id')))
    attire_counts = list(profiles_qs.values('attire').annotate(count=Count('id')))

    camera_stats = Camera.objects.aggregate(
        total_cameras=Count('id'),
        active_cameras=Count('id', filter=Q(status='online')),
    )

    total_visit_duration = stats['total_dwell'] or 0
    avg_dwell = round(stats['avg_dwell'] or 0, 1)
    peak_hour = max(visits_by_hour, key=lambda x: x['count'])['hour'] if visits_by_hour else None

    result = {
        'total_visits': stats['total_visits'] or 0,
        'visits_by_date': visits_by_date,
        'visits_by_camera': visits_by_camera,
        'visits_by_hour': visits_by_hour,
        'gender_counts': gender_counts,
        'attire_counts': attire_counts,
        'avg_dwell_time': avg_dwell,
        'total_dwell_time': total_visit_duration,
        'new_persons': person_stats['new_persons'] or 0,
        'peak_hour': peak_hour,
        'total_cameras': camera_stats['total_cameras'] or 0,
        'active_cameras': camera_stats['active_cameras'] or 0,
    }
    SERVICE_CACHE[cache_key] = {'timestamp': time.time(), 'data': result}
    return result


def _chart_to_image(drawing, width=4.5 * inch, height=2.5 * inch):
    """Convert a reportlab Drawing to a PDF image, or return None if rendering fails."""
    try:
        png_bytes = renderPM.drawToString(drawing, fmt='PNG')
        return ReportImage(BytesIO(png_bytes), width=width, height=height)
    except Exception:
        # If PNG rasterization fails (platform rendering issues), fall back to
        # returning the Drawing itself which Platypus can render as a vector
        # graphic. This ensures charts appear in the PDF even when renderPM
        # cannot produce a PNG.
        try:
            # Try to set width/height hints on the drawing for nicer layout.
            drawing.width = width
            drawing.height = height
        except Exception:
            pass
        return drawing


def _visits_by_date_chart(visits_by_date):
    counts = [v['count'] for v in visits_by_date] or [0]
    labels = [v['date'] for v in visits_by_date] or ['No data']
    max_val = max(counts) or 1
    d = Drawing(450, 250)
    lc = HorizontalLineChart()
    lc.x = 50
    lc.y = 50
    lc.height = 150
    lc.width = 350
    lc.data = [counts]
    lc.categoryAxis.categoryNames = labels
    lc.categoryAxis.labels.angle = 45
    lc.categoryAxis.labels.fontSize = 8
    lc.valueAxis.valueMin = 0
    lc.valueAxis.valueMax = int(max_val * 1.2) or 1
    lc.lines[0].strokeColor = colors.HexColor('#4f46e5')
    lc.lines[0].strokeWidth = 2
    d.add(lc)
    return d


def _visits_by_camera_chart(visits_by_camera):
    names = [c['camera__name'][:14] for c in visits_by_camera] or ['No data']
    counts = [c['count'] for c in visits_by_camera] or [0]
    max_val = max(counts) or 1
    d = Drawing(450, 250)
    bc = VerticalBarChart()
    bc.x = 50
    bc.y = 50
    bc.height = 150
    bc.width = 350
    bc.data = [counts]
    bc.categoryAxis.categoryNames = names
    bc.categoryAxis.labels.fontSize = 8
    bc.categoryAxis.labels.angle = 30
    bc.valueAxis.valueMin = 0
    bc.valueAxis.valueMax = int(max_val * 1.2) or 1
    bc.bars[0].fillColor = colors.HexColor('#14b8a6')
    d.add(bc)
    return d


def _visits_by_hour_chart(visits_by_hour):
    counts = [h['count'] for h in visits_by_hour]
    max_val = max(counts) or 1
    d = Drawing(450, 250)
    bc = HorizontalBarChart()
    bc.x = 60
    bc.y = 30
    bc.height = 170
    bc.width = 340
    bc.data = [counts]
    bc.categoryAxis.categoryNames = [str(h['hour']) for h in visits_by_hour]
    bc.categoryAxis.labels.fontSize = 7
    bc.valueAxis.valueMin = 0
    bc.valueAxis.valueMax = int(max_val * 1.2) or 1
    bc.bars[0].fillColor = colors.HexColor('#f59e0b')
    d.add(bc)
    return d


def _pie_chart(items, label_key, value_key, label_map):
    counts = [item[value_key] for item in items] or [1]
    labels = [label_map.get(item.get(label_key), str(item.get(label_key))) for item in items] or ['No data']
    d = Drawing(300, 200)
    pc = Pie()
    pc.x = 65
    pc.y = 15
    pc.width = 170
    pc.height = 170
    pc.data = counts
    pc.labels = labels
    pc.slices.strokeWidth = 0.5
    d.add(pc)
    return d


@login_required
def analytics(request):
    """Analytics dashboard with charts and filters."""
    start_date, end_date, range_label = _parse_analytics_date_range(request)
    data = _get_analytics_data(start_date, end_date)
    data['range_label'] = range_label
    data['start_date'] = start_date.strftime('%Y-%m-%d')
    data['end_date'] = end_date.strftime('%Y-%m-%d')
    data['selected_range'] = request.GET.get('range', '7days')
    data['gender_labels'] = _GENDER_LABELS
    data['attire_labels'] = _ATTIRE_LABELS
    return render(request, 'core/analytics.html', data)


@login_required
def export_analytics_pdf(request):
    """Generate a PDF analytics report for the selected date range."""
    start_date, end_date, range_label = _parse_analytics_date_range(request)
    data = _get_analytics_data(start_date, end_date)

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'TitleCustom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        'SubtitleCustom',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#475569'),
        spaceAfter=12,
    )
    heading2 = ParagraphStyle(
        'Heading2Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=8,
        spaceBefore=10,
    )
    body_style = ParagraphStyle(
        'BodyCustom',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9,
        leading=11,
        textColor=colors.HexColor('#334155'),
    )
    elements = []

    elements.append(Paragraph('Analytics Report', title_style))
    elements.append(Paragraph(
        f'Period: {range_label} | Generated: {timezone.now().strftime("%Y-%m-%d %H:%M:%S")}',
        subtitle_style,
    ))

    summary_data = [
        ['Metric', 'Value'],
        ['Total Visits', str(data['total_visits'])],
        ['New Persons', str(data['new_persons'])],
        ['Average Dwell Time', f"{data['avg_dwell_time']}s"],
        ['Total Dwell Time', f"{data['total_dwell_time']}s"],
        ['Peak Hour', f"{data['peak_hour']:02d}:00" if data['peak_hour'] is not None else 'N/A'],
        ['Cameras', str(data['total_cameras'])],
        ['Online Cameras', str(data['active_cameras'])],
    ]
    summary_table = Table(summary_data, colWidths=[2.2 * inch, 2.1 * inch], hAlign='LEFT')
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4f46e5')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fbff')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dfe7f1')),
        ('FONTSIZE', (0, 1), (-1, -1), 8.5),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#ffffff'), colors.HexColor('#f8fbff')]),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 0.12 * inch))

    highlight_text = (
        f"Peak traffic hour: {data['peak_hour']:02d}:00" if data['peak_hour'] is not None else 'Peak traffic hour: N/A'
    ) + f" • Total cameras: {data['total_cameras']} • Online cameras: {data['active_cameras']}"
    elements.append(Paragraph(highlight_text, body_style))
    elements.append(Spacer(1, 0.16 * inch))

    elements.append(Paragraph('Traffic Overview', heading2))
    chart_row = []
    visits_chart = _chart_to_image(_visits_by_date_chart(data['visits_by_date']), width=3.7 * inch, height=2.2 * inch)
    camera_chart = _chart_to_image(_visits_by_camera_chart(data['visits_by_camera']), width=3.7 * inch, height=2.2 * inch)
    chart_row.append(visits_chart or Paragraph('Visits over time chart unavailable.', body_style))
    chart_row.append(camera_chart or Paragraph('Camera chart unavailable.', body_style))
    chart_table = Table([chart_row], colWidths=[3.95 * inch, 3.95 * inch], hAlign='LEFT')
    chart_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    elements.append(chart_table)
    elements.append(Spacer(1, 0.12 * inch))

    elements.append(Paragraph('Traffic Distribution', heading2))
    hour_chart = _chart_to_image(_visits_by_hour_chart(data['visits_by_hour']), width=3.7 * inch, height=2.2 * inch)
    gender_items = [{'gender': g['gender'], 'count': g['count']} for g in data['gender_counts']]
    attire_items = [{'attire': a['attire'], 'count': a['count']} for a in data['attire_counts']]
    gender_chart = _chart_to_image(_pie_chart(gender_items, 'gender', 'count', _GENDER_LABELS), width=3.2 * inch, height=2.0 * inch)
    attire_chart = _chart_to_image(_pie_chart(attire_items, 'attire', 'count', _ATTIRE_LABELS), width=3.2 * inch, height=2.0 * inch)

    distribution_row = [hour_chart or Paragraph('Hourly chart unavailable.', body_style), gender_chart or Paragraph('Gender chart unavailable.', body_style)]
    distribution_table = Table([distribution_row], colWidths=[3.95 * inch, 3.95 * inch], hAlign='LEFT')
    distribution_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    elements.append(distribution_table)
    elements.append(Spacer(1, 0.12 * inch))
    elements.append(Paragraph('Attire Segmentation', heading2))
    if attire_chart:
        elements.append(attire_chart)
    else:
        elements.append(Paragraph('Attire chart unavailable.', body_style))

    # Header/footer drawing callback
    def _draw_header_footer(canvas_obj, doc_obj):
        canvas_obj.saveState()
        width, height = doc_obj.pagesize

        # Header: colored bar with small title and optional logo placeholder
        header_height = 36
        canvas_obj.setFillColor(colors.HexColor('#4f46e5'))
        canvas_obj.rect(0, height - header_height, width, header_height, stroke=0, fill=1)
        canvas_obj.setFillColor(colors.white)
        canvas_obj.setFont('Helvetica-Bold', 12)
        canvas_obj.drawString(40, height - header_height + 9, 'Analytics Report')
        canvas_obj.setFont('Helvetica', 8)
        canvas_obj.drawRightString(width - 40, height - header_height + 11, timezone.now().strftime('%Y-%m-%d %H:%M:%S'))

        # Footer: generation note and page number
        footer_y = 28
        canvas_obj.setFillColor(colors.HexColor('#94a3b8'))
        canvas_obj.setFont('Helvetica', 8)
        canvas_obj.drawString(40, footer_y, f'Generated by VMS on {timezone.now().strftime("%Y-%m-%d %H:%M:%S")}')
        canvas_obj.drawRightString(width - 40, footer_y, f'Page {doc_obj.page}')
        canvas_obj.restoreState()

    # Add legends for pie charts to improve readability when charts are small
    def _legend_table(items, label_map):
        rows = [[ 'Label', 'Count' ]]
        for it in items:
            label = label_map.get(it.get(list(it.keys())[0]), str(it.get(list(it.keys())[0])))
            count = it.get(list(it.keys())[1], it.get('count'))
            rows.append([label, str(count)])
        t = Table(rows, colWidths=[2.0 * inch, 1.0 * inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#eef2ff')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.HexColor('#e2e8f0')),
        ]))
        return t

    # Insert legends after the corresponding charts (if data exists)
    if data.get('gender_counts'):
        gender_legend = _legend_table(gender_items, _GENDER_LABELS)
        elements.append(Spacer(1, 0.06 * inch))
        elements.append(gender_legend)

    if data.get('attire_counts'):
        attire_legend = _legend_table(attire_items, _ATTIRE_LABELS)
        elements.append(Spacer(1, 0.06 * inch))
        elements.append(attire_legend)

    # Build document with header/footer callbacks
    doc.build(elements, onFirstPage=_draw_header_footer, onLaterPages=_draw_header_footer)
    pdf = buffer.getvalue()
    buffer.close()
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="analytics-report-{start_date}-{end_date}.pdf"'
    response.write(pdf)
    return response


@login_required
def add_camera(request):
    if request.method == 'POST':
        form = CameraForm(request.POST)
        if form.is_valid():
            camera = form.save()
            channel_cameras = [camera]
            channel_urls = _rtsp_channel_urls(camera.source)
            if channel_urls:
                camera.source = channel_urls[0]
                camera.name = f'{camera.name} - Channel 1'
                camera.save(update_fields=['name', 'source', 'updated_at'])
                camera_values = {
                    field.name: getattr(camera, field.name)
                    for field in Camera._meta.concrete_fields
                    if field.name not in {'id', 'name', 'source', 'created_at', 'updated_at'}
                }
                for channel, channel_url in enumerate(channel_urls[1:], start=2):
                    channel_cameras.append(Camera.objects.create(
                        **camera_values,
                        name=f'{form.cleaned_data["name"]} - Channel {channel}',
                        source=channel_url,
                    ))

            for channel_camera in channel_cameras:
                service = get_service(channel_camera.pk)
                try:
                    service.start()
                    channel_camera.refresh_from_db()
                except Exception as exc:
                    channel_camera.status = 'error'
                    channel_camera.last_error = str(exc)
                    channel_camera.save(update_fields=['status', 'last_error', 'updated_at'])

            return redirect('core:cameras')

        camera_list = _get_camera_list_data()
        return render(request, 'core/cameras.html', {'cameras': camera_list, 'form': form})

    return redirect('core:cameras')


@login_required
def edit_camera(request, pk):
    camera = get_object_or_404(Camera, pk=pk)
    if request.method == 'POST':
        form = CameraForm(request.POST, instance=camera)
        if form.is_valid():
            cam = form.save(commit=False)
            cam.last_error = ''
            cam.save()
            if pk in SERVICE_CACHE:
                try:
                    SERVICE_CACHE[pk].stop()
                    if cam.enabled:
                        SERVICE_CACHE[pk].start()
                except Exception:
                    pass
            return redirect('core:cameras')
        camera_list = _get_camera_list_data()
        return render(request, 'core/cameras.html', {'cameras': camera_list, 'form': form, 'editing_camera': camera})

    form = CameraForm(instance=camera)
    camera_list = _get_camera_list_data()
    return render(request, 'core/cameras.html', {'cameras': camera_list, 'form': form, 'editing_camera': camera})


@login_required
@require_POST
def delete_camera(request, pk):
    camera = get_object_or_404(Camera, pk=pk)
    if pk in SERVICE_CACHE:
        try:
            SERVICE_CACHE[pk].stop()
            del SERVICE_CACHE[pk]
        except Exception:
            pass
    camera.delete()
    return redirect('core:cameras')




@login_required
def live_camera(request, pk):
    camera = get_object_or_404(Camera, pk=pk)
    active_tab = request.GET.get('tab', 'live')
    if active_tab not in {'live', 'recordings', 'profiles'}:
        active_tab = 'live'

    from django.core.cache import cache
    has_live_feed = bool(cache.get(f'vms:live_frame:{pk}'))
    service = get_service(pk)
    if active_tab == 'live' and not has_live_feed and not (service._thread and service._thread.is_alive()):
        service.start()
    state = service.get_camera_state(camera)
    if has_live_feed:
        state['status'] = 'online'
        state['last_error'] = ''
    recent_videos = Video.objects.filter(camera=camera, ready=True).order_by('-created_at')[:25]
    videos = [video for video in recent_videos if _is_playable_mp4(video.file_path)][:10]

    selected_video_id = request.GET.get('video_id')
    selected_profile_video_id = request.GET.get('profile_video_id')
    selected_video = None

    if selected_video_id:
        candidate = get_object_or_404(Video, pk=selected_video_id, camera=camera)
        if _is_playable_mp4(candidate.file_path):
            selected_video = candidate
    elif selected_profile_video_id:
        profile = get_object_or_404(VisitProfile, pk=selected_profile_video_id, camera=camera)
        if profile.video and _is_playable_mp4(profile.video.file_path):
            selected_video = profile.video

    # Pre-compute human-readable attire labels for visit profiles.
    visit_profiles = VisitProfile.objects.filter(camera=camera).order_by('-counted_time')[:25]
    for profile in visit_profiles:
        profile.attire_display = profile.get_attire_display()
        profile.attire_summary = _format_attire_summary(profile.attire_attributes)

    return render(
        request,
        'core/live.html',
        {
            'camera': camera,
            'state': state,
            'visitors': Visitor.objects.filter(camera=camera).order_by('-counted_time')[:10],
            'visit_profiles': visit_profiles,
            'videos': videos,
            'active_tab': active_tab,
            'selected_video': selected_video,
        },
    )


@login_required
def camera_state(request, pk):
    camera = get_object_or_404(Camera, pk=pk)
    from django.core.cache import cache
    has_live_feed = bool(cache.get(f'vms:live_frame:{pk}'))
    service = get_service(pk)
    if not has_live_feed and not (service._thread and service._thread.is_alive()):
        service.start()
    state = service.get_camera_state(camera)
    if has_live_feed:
        state['status'] = 'online'
        state['last_error'] = ''
    return JsonResponse({
        'status': state['status'],
        'recording': state['recording'],
        'motion_detected': state['motion_detected'],
        'recording_duration_display': state['recording_duration_display'],
        'active_tracks': state['active_tracks'],
        'zone_occupancy': state['zone_occupancy'],
        'queue_max_capacity': state['queue_max_capacity'],
        'queue_overcrowded': state['queue_overcrowded'],
        'tripwire_in': state['tripwire_in'],
        'tripwire_out': state['tripwire_out'],
        'visitor_count': state['visitor_count'],
        'analytics': state['analytics'],
        'last_error': state['last_error'],
        'source_type': camera.source_type,
        'zone_enabled': camera.zone_enabled,
        'tripwire_enabled': camera.tripwire_enabled,
    })


@login_required
def camera_stream(request, pk):
    from django.core.cache import cache
    camera = get_object_or_404(Camera, pk=pk)
    service = get_service(pk)

    def generate():
        frame_file = os.path.join(settings.MEDIA_ROOT, 'live_frames', f'camera_{pk}.jpg')
        last_sent = None
        while True:
            # 1. Direct stream of pre-encoded JPEG bytes from camera service
            jpeg_bytes = service.get_latest_jpeg()
            if jpeg_bytes is not None:
                if jpeg_bytes != last_sent:
                    last_sent = jpeg_bytes
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
            else:
                # 2. Fallback to latest RGB frame and encode
                frame = service.get_latest_frame()
                if frame is not None:
                    success, encoded = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    if success:
                        last_sent = encoded.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + last_sent + b'\r\n')
                else:
                    cached_jpg = cache.get(f'vms:live_frame:{pk}')
                    if cached_jpg:
                        if cached_jpg != last_sent:
                            last_sent = cached_jpg
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + cached_jpg + b'\r\n')
                    elif os.path.exists(frame_file):
                        try:
                            with open(frame_file, 'rb') as handle:
                                file_jpg = handle.read()
                            if file_jpg and file_jpg != last_sent:
                                last_sent = file_jpg
                                yield (b'--frame\r\n'
                                       b'Content-Type: image/jpeg\r\n\r\n' + file_jpg + b'\r\n')
                        except Exception:
                            pass
                    elif not service.start():
                        time.sleep(0.1)
            time.sleep(0.04)

    response = StreamingHttpResponse(generate(), content_type='multipart/x-mixed-replace; boundary=frame')
    response['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    response['Pragma'] = 'no-cache'
    response['Expires'] = '0'
    response['X-Accel-Buffering'] = 'no'
    return response


@login_required
def recordings(request, pk):
    camera = get_object_or_404(Camera, pk=pk)
    return redirect(f"{reverse('core:live_camera', args=[camera.pk])}?tab=recordings")


@login_required
def play_recording(request, pk):
    video = get_object_or_404(Video, pk=pk)
    if not video.ready:
        return HttpResponse('Recording is still in progress and cannot be played yet.', status=409)
    return _render_simple_page(
        'Recording Playback',
        f"<h1>Recording Playback</h1><p>Camera: {escape(video.camera.name)}</p><p>Duration: {escape(str(video.duration_seconds))}s</p><video controls autoplay><source src='{reverse('core:stream_recording', args=[video.pk])}' type='video/mp4'></video>",
    )


@login_required
def stream_recording(request, pk):
    video = get_object_or_404(Video, pk=pk)
    if not video.ready:
        return HttpResponse('Recording is still in progress and cannot be streamed yet.', status=409)

    if _is_playable_mp4(video.file_path):
        return _video_file_response(request, video.file_path, f'recording-{video.pk}.mp4')

    clip_dir = os.path.join(settings.MEDIA_ROOT, 'tmp')
    os.makedirs(clip_dir, exist_ok=True)
    clip_path = os.path.join(clip_dir, f'recording-{video.pk}.mp4')
    if os.path.exists(clip_path):
        try:
            os.remove(clip_path)
        except OSError:
            pass

    if os.path.exists(video.file_path):
        try:
            cap = cv2.VideoCapture(video.file_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
                target_frames = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1, max(1, 120 * fps)))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
                if writer.isOpened():
                    for _ in range(target_frames):
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            break
                        writer.write(frame)
                    writer.release()
                    if os.path.exists(clip_path):
                        transcode_to_h264(clip_path)
                cap.release()
        except Exception:
            pass

    if not os.path.exists(clip_path):
        if not _create_fallback_mp4(clip_path, label='Recording Preview'):
            return HttpResponse('Recording not found', status=404)

    if not os.path.exists(clip_path) or os.path.getsize(clip_path) <= 0:
        return HttpResponse('Recording not found', status=404)

    return _video_file_response(request, clip_path, f'recording-{video.pk}.mp4')


def _format_attire_summary(attributes):
    """Return a short, human-readable attire attribute summary."""
    if not attributes:
        return ''
    parts = []
    coverage = attributes.get('coverage')
    if coverage:
        parts.append(coverage.replace('_', ' '))
    sleeve = attributes.get('sleeve_length')
    if sleeve:
        parts.append(sleeve.replace('_', ' '))
    fit = attributes.get('fit')
    if fit:
        parts.append(fit)
    colour = attributes.get('dominant_colour')
    if colour and colour != 'unknown':
        parts.append(colour)
    return ', '.join(parts) if parts else ''


def _get_visit_profiles_for_export(camera, date_filter):
    profiles = VisitProfile.objects.filter(camera=camera).order_by('-counted_time')
    if date_filter:
        try:
            from datetime import date as dt_date
            parsed = dt_date.fromisoformat(date_filter)
            profiles = profiles.filter(date=parsed)
        except ValueError:
            pass
    return profiles


@login_required
def export_visit_profiles(request, pk):
    """Export visit profiles for a camera as CSV or PDF."""
    camera = get_object_or_404(Camera, pk=pk)
    date_filter = request.GET.get('date')
    export_format = request.GET.get('format', 'csv').lower()
    profiles = _get_visit_profiles_for_export(camera, date_filter)

    if export_format == 'pdf':
        return _export_visit_profiles_pdf(request, camera, profiles)

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="visit-profiles-{camera.slug}-{timezone.now().date()}.csv"'
    writer = csv.writer(response)
    writer.writerow([
        'Sequence', 'Track ID', 'Person Profile ID', 'Date', 'Entry Time',
        'Counted Time', 'Required Seconds', 'Dwell Time', 'Gender', 'Attire',
        'Attire Summary', 'Pose', 'Liveness Score', 'Snapshot Path', 'Video Path',
    ])
    for profile in profiles:
        writer.writerow([
            profile.sequence_number,
            profile.track_id,
            profile.person_profile_id,
            profile.date,
            profile.entry_time,
            profile.counted_time,
            profile.required_seconds,
            profile.dwell_time,
            profile.get_gender_display(),
            profile.get_attire_display(),
            _format_attire_summary(profile.attire_attributes),
            profile.pose or 'unknown',
            f"{profile.liveness_score:.2f}",
            profile.snapshot_path,
            profile.video.file_path if profile.video else '',
        ])
    return response


def _export_visit_profiles_pdf(request, camera, profiles):
    """Generate a PDF report of visit profiles with embedded snapshot images."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        topMargin=0.8 * inch,
        bottomMargin=0.8 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'TitleCustom',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#4f46e5'),
        spaceAfter=12,
    )
    elements = []

    elements.append(Paragraph(f'Visit Profiles - {camera.name}', title_style))
    elements.append(Paragraph(
        f'Generated: {timezone.now().strftime("%Y-%m-%d %H:%M:%S")} | Total profiles: {profiles.count()}',
        styles['Normal'],
    ))
    elements.append(Spacer(1, 0.2 * inch))

    data = [[
        'Visit #', 'Person', 'Date', 'Entry', 'Counted',
        'Required', 'Dwell', 'Gender', 'Attire', 'Pose', 'Live',
    ]]
    for profile in profiles:
        data.append([
            str(profile.sequence_number),
            f"#{profile.person_profile_id or 'new'}",
            profile.date.strftime('%Y-%m-%d'),
            profile.entry_time.strftime('%H:%M:%S'),
            profile.counted_time.strftime('%H:%M:%S'),
            f"{profile.required_seconds}s",
            f"{profile.dwell_time}s",
            profile.get_gender_display(),
            profile.get_attire_display(),
            profile.pose or 'unknown',
            f"{profile.liveness_score:.2f}",
        ])

    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4f46e5')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fbff')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dfe7f1')),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 0.3 * inch))

    # Add video evidence section.
    video_profiles = [p for p in profiles if p.video]
    if video_profiles:
        elements.append(Paragraph('Video Evidence', styles['Heading2']))
        elements.append(Spacer(1, 0.1 * inch))
        for profile in video_profiles:
            attire_text = profile.get_attire_display()
            if profile.attire_attributes:
                attire_summary = _format_attire_summary(profile.attire_attributes)
                if attire_summary:
                    attire_text = f'{attire_text} — {attire_summary}'
            elements.append(Paragraph(
                f'Visit #{profile.sequence_number} — {profile.date} {profile.counted_time.strftime("%H:%M:%S")} '
                f'({profile.get_gender_display()}, {attire_text}, {profile.pose or "unknown"})',
                styles['Normal'],
            ))
            video_url = request.build_absolute_uri(reverse('core:stream_recording', args=[profile.video.pk]))
            elements.append(Paragraph(f'<a href="{video_url}" color="blue">View video</a>', styles['Normal']))
            elements.append(Spacer(1, 0.15 * inch))

    doc.build(elements)
    pdf = buffer.getvalue()
    buffer.close()
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="visit-profiles-{camera.slug}-{timezone.now().date()}.pdf"'
    response.write(pdf)
    return response


def get_service(camera_id):
    if camera_id not in SERVICE_CACHE:
        SERVICE_CACHE[camera_id] = CameraService(camera_id)
    return SERVICE_CACHE[camera_id]


@login_required
@require_POST
def update_camera_geometry(request, pk):
    """API endpoint to update zone & tripwire percentage coordinates from interactive canvas."""
    import json
    camera = get_object_or_404(Camera, pk=pk)
    try:
        data = json.loads(request.body)
        if 'zone_enabled' in data:
            camera.zone_enabled = bool(data['zone_enabled'])
        if 'zone_x_min' in data:
            camera.zone_x_min = int(data['zone_x_min'])
        if 'zone_y_min' in data:
            camera.zone_y_min = int(data['zone_y_min'])
        if 'zone_x_max' in data:
            camera.zone_x_max = int(data['zone_x_max'])
        if 'zone_y_max' in data:
            camera.zone_y_max = int(data['zone_y_max'])

        if 'tripwire_enabled' in data:
            camera.tripwire_enabled = bool(data['tripwire_enabled'])
        if 'tripwire_x1' in data:
            camera.tripwire_x1 = int(data['tripwire_x1'])
        if 'tripwire_y1' in data:
            camera.tripwire_y1 = int(data['tripwire_y1'])
        if 'tripwire_x2' in data:
            camera.tripwire_x2 = int(data['tripwire_x2'])
        if 'tripwire_y2' in data:
            camera.tripwire_y2 = int(data['tripwire_y2'])

        if 'queue_max_capacity' in data:
            camera.queue_max_capacity = max(1, int(data['queue_max_capacity']))

        camera.save()
        if pk in SERVICE_CACHE:
            try:
                SERVICE_CACHE[pk]._camera.refresh_from_db()
            except Exception:
                pass
        return JsonResponse({
            'status': 'ok',
            'message': 'Camera geometry updated successfully.',
            'zone_enabled': camera.zone_enabled,
            'tripwire_enabled': camera.tripwire_enabled,
        })
    except Exception as exc:
        return JsonResponse({'status': 'error', 'message': str(exc)}, status=400)


@login_required
def search_by_photo_view(request):
    """Search PersonProfile records by uploading a target photo."""
    from .services.search_service import get_search_service

    results = []
    searched = False
    if request.method == 'POST' and request.FILES.get('photo'):
        searched = True
        photo = request.FILES['photo']
        image_bytes = photo.read()
        search_service = get_search_service()
        results = search_service.search_profiles_by_image(image_bytes)

    return render(request, 'core/search.html', {
        'searched': searched,
        'results': results,
    })


@login_required
@require_POST
def edit_person_profile(request, pk):
    """Update name, VIP/Blacklist tag, and notes for a PersonProfile."""
    profile = get_object_or_404(PersonProfile, pk=pk)
    name = request.POST.get('name', '').strip()
    person_type = request.POST.get('person_type', 'regular')
    notes = request.POST.get('notes', '').strip()

    profile.name = name
    if person_type in ('regular', 'vip', 'blacklist'):
        profile.person_type = person_type
    profile.notes = notes
    profile.save()

    next_url = request.POST.get('next') or request.META.get('HTTP_REFERER') or reverse('core:dashboard')
    return redirect(next_url)
