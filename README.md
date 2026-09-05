# Aura VMS — Production Deployment Guide

Enterprise Vision Monitoring System (VMS) with real-time RTSP stream processing, YOLOv8 person tracking, counting zones, tripwire crossing analytics, and visit profile identification.

---

## 🌐 Production Server Specifications

- **Domain Name**: `vms.lovosis.in`
- **Server Public IP**: `72.61.246.121`
- **Application Port**: `8000` (Internal Gunicorn WSGI binding to `127.0.0.1:8000`)
- **Public Web Ports**: `80` (HTTP → HTTPS redirect) and `443` (HTTPS with SSL)
- **Deployment Directory**: `/opt/vms`

---

## 🏗 System Architecture Overview

```
                          [ Internet / Browser Client ]
                                        │
                                        ▼ HTTPS (443)
                         ┌─────────────────────────────┐
                         │   Nginx Web Server Proxy    │
                         │      (vms.lovosis.in)       │
                         └──────────────┬──────────────┘
                                        │ Proxy Pass (127.0.0.1:8000)
                                        │ (Zero-Buffering for RTSP)
                                        ▼
                         ┌─────────────────────────────┐
                         │     Gunicorn WSGI App       │
                         │     (2 Workers, 4 Threads)  │
                         └──────────────┬──────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
             ┌───────────────────────┐     ┌───────────────────────┐
             │ SQLite / PostgreSQL   │     │ Redis / File Cache    │
             │   (Database State)    │     │  (Live JPEG Frames)   │
             └───────────────────────┘     └───────────▲───────────┘
                                                       │
                                   Frame Updates (15Hz)│
                                                       │
                         ┌─────────────────────────────┴───────────┐
                         │   Worker Supervisor (manage_workers)    │
                         │   Auto-spawns run_camera_worker <id>    │
                         │    Continuous 24/7 YOLOv8 Tracking      │
                         └─────────────────▲───────────────────────┘
                                           │ RTSP / TCP
                                           │
                         ┌─────────────────┴───────────┐
                         │      IP / RTSP Cameras      │
                         │  (e.g., Port 554 / 6000)    │
                         └─────────────────────────────┘
```

---

## 🚀 Step-by-Step Production Hosting Guide

### Step 1: Server Prerequisites & System Dependencies

Connect to your VPS (`ssh root@72.61.246.121`) and install the required system libraries:

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python 3.10+, pip, venv, Git, Nginx, Supervisor, and media libraries
sudo apt install -y python3 python3-pip python3-venv git nginx supervisor \
    ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 redis-server certbot python3-certbot-nginx
```

Verify Redis is running (recommended for ultra-low latency frame caching):
```bash
sudo systemctl enable --now redis-server
sudo systemctl status redis-server
```

---

### Step 2: Clone and Setup Application Directory

Place the project in `/opt/vms`:

```bash
# Create directory and set permissions
sudo mkdir -p /opt/vms
sudo chown -R www-data:www-data /opt/vms

# If copying from local machine or Git:
cd /opt/vms
# (Extract or git clone code here)

# Create Python Virtual Environment
python3 -m venv /opt/vms/venv
source /opt/vms/venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install gunicorn
```

---

### Step 3: Environment Configuration (`.env`)

Create or update `/opt/vms/.env`:

```bash
sudo nano /opt/vms/.env
```

Paste the following production configuration:

```env
# Django Core Settings
SECRET_KEY=b9v%+^j6i(6%si^fe98x1+=9z)9!vt9n=z6p408drfj_(6r8wy
DEBUG=False
ALLOWED_HOSTS=vms.lovosis.in,72.61.246.121,localhost,127.0.0.1

# HTTPS & Security
SECURE_SSL_REDIRECT=True
SESSION_COOKIE_SECURE=True
CSRF_COOKIE_SECURE=True
SECURE_HSTS_SECONDS=31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS=True
SECURE_HSTS_PRELOAD=True
SECURE_PROXY_SSL_HEADER=HTTP_X_FORWARDED_PROTO,https
SECURE_CONTENT_TYPE_NOSNIFF=True
SECURE_REFERRER_POLICY=strict-origin-when-cross-origin

# Shared Cache (Redis for multi-process performance)
REDIS_URL=redis://127.0.0.1:6379/1

# Media Storage
MEDIA_ROOT=/opt/vms/media
STATIC_ROOT=/opt/vms/staticfiles
```

Save and exit (`Ctrl+O`, `Enter`, `Ctrl+X`).

---

### Step 4: Run Migrations and Collect Static Files

```bash
cd /opt/vms
source /opt/vms/venv/bin/activate

# Apply database migrations
python manage.py migrate

# Collect static files into STATIC_ROOT
python manage.py collectstatic --noinput

# Create an administrator account
python manage.py createsuperuser

# Ensure correct permissions for www-data
sudo chown -R www-data:www-data /opt/vms
sudo chmod -R 775 /opt/vms/media /opt/vms/.cache
```

---

### Step 5: Configure Supervisor (24/7 Web & Camera Workers)

Supervisor ensures both Gunicorn and the camera worker supervisor run continuously and restart automatically on reboot or crash.

Verify `/opt/vms/vms_supervisor.conf`:

```ini
[program:vms_web]
command=/opt/vms/venv/bin/gunicorn config.wsgi:application --bind 127.0.0.1:8000 --workers 2 --threads 4 --timeout 120
directory=/opt/vms
user=www-data
autostart=true
autorestart=true
stopasgroup=true
killasgroup=true
stderr_logfile=/var/log/vms_web.err.log
stdout_logfile=/var/log/vms_web.out.log
environment=PYTHONUNBUFFERED="1"

[program:vms_worker_supervisor]
command=/opt/vms/venv/bin/python manage.py manage_workers --poll-interval 5
directory=/opt/vms
user=www-data
autostart=true
autorestart=true
stopasgroup=true
killasgroup=true
stderr_logfile=/var/log/vms_supervisor.err.log
stdout_logfile=/var/log/vms_supervisor.out.log
environment=PYTHONUNBUFFERED="1"
```

Link the configuration to Supervisor and start services:

```bash
# Link configuration
sudo cp /opt/vms/vms_supervisor.conf /etc/supervisor/conf.d/vms.conf

# Reload Supervisor
sudo supervisorctl reread
sudo supervisorctl update

# Check status
sudo supervisorctl status
```

You should see:
```
vms_web                  RUNNING   pid 1234, uptime 0:01:00
vms_worker_supervisor    RUNNING   pid 1235, uptime 0:01:00
```

---

### Step 6: Configure Nginx Reverse Proxy with Zero-Buffering

Create `/etc/nginx/sites-available/vms`:

```bash
sudo nano /etc/nginx/sites-available/vms
```

Paste the following configuration (critical: `proxy_buffering off` for real-time live streaming):

```nginx
server {
    listen 80;
    server_name vms.lovosis.in 72.61.246.121;

    # Client body limit for video / image uploads
    client_max_body_size 100M;

    # Static files
    location /static/ {
        alias /opt/vms/staticfiles/;
        expires 30d;
        add_header Cache-Control "public, no-transform";
    }

    # Media files (snapshots, recorded mp4 videos)
    location /media/ {
        alias /opt/vms/media/;
        expires 7d;
        add_header Cache-Control "public, no-transform";
    }

    # Camera Live Streams — Zero Buffering for instantaneous playback
    location ~* ^/camera/\d+/stream/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Disable all proxy buffering for live MJPEG feeds
        proxy_buffering off;
        proxy_cache off;
        chunked_transfer_encoding off;
        proxy_read_timeout 86400s;
        proxy_send_timeout 86400s;
    }

    # Standard Application Routes
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
        proxy_connect_timeout 60s;
    }
}
```

Enable the Nginx site and test configuration:

```bash
# Enable site
sudo ln -sf /etc/nginx/sites-available/vms /etc/nginx/sites-enabled/

# Test Nginx syntax
sudo nginx -t

# Reload Nginx
sudo systemctl reload nginx
```

---

### Step 7: Secure with Free SSL Certificate (Let's Encrypt)

Run Certbot to obtain and install a free SSL certificate for `vms.lovosis.in`:

```bash
sudo certbot --nginx -d vms.lovosis.in
```

Choose **Option 2** (Redirect HTTP traffic to HTTPS). Certbot will automatically configure HTTPS on port 443 with auto-renewing certificates.

Verify auto-renewal:
```bash
sudo certbot renew --dry-run
```

---

## 🛠 Useful Operations & Maintenance Commands

### Manage Services via Supervisor
```bash
# Check status of web and camera workers
sudo supervisorctl status

# Restart web application
sudo supervisorctl restart vms_web

# Restart camera worker supervisor
sudo supervisorctl restart vms_worker_supervisor

# Restart everything
sudo supervisorctl restart all
```

### View Live Logs
```bash
# Gunicorn Web Access / Error Logs
sudo tail -f /var/log/vms_web.err.log
sudo tail -f /var/log/vms_web.out.log

# Camera Worker AI Analytics Logs
sudo tail -f /var/log/vms_supervisor.err.log
sudo tail -f /var/log/vms_supervisor.out.log

# Nginx Logs
sudo tail -f /var/log/nginx/error.log
sudo tail -f /var/log/nginx/access.log
```

---

## 🔍 Troubleshooting Camera Disconnections

If a camera shows as disconnected on the dashboard:
1. **RTSP Stream Disconnected**:
   - Verify camera network reachability from the server:
     ```bash
     ping -c 3 72.61.246.121
     nc -zv 72.61.246.121 6000
     ```
   - Verify RTSP URL credentials: `rtsp://admin:password@72.61.246.121:6000/media/video2`.
   - Ensure the IP camera has not reached its maximum concurrent RTSP client limit.
2. **High VPS CPU Usage**:
   - The system is pre-configured with `imgsz=416` and 2-frame tracking cadence to keep CPU load low.
   - You can verify current CPU and memory consumption anytime with `htop` or `top`.
