#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
release_dir="$1"
release_sha="$2"
apt-get update -qq
apt-get install -y python3.12 python3.12-venv nginx libgl1 libglib2.0-0t64
systemctl stop nginx
id opendart >/dev/null 2>&1 || useradd --system --home-dir /var/lib/opendart --shell /usr/sbin/nologin opendart
install -d -o opendart -g opendart /var/lib/opendart/jobs /var/lib/opendart/validation /var/lib/opendart/matplotlib /var/log/opendart
python3.12 -m venv /opt/opendart/venv
/opt/opendart/venv/bin/python -m pip install --only-binary=:all: -r "$release_dir/deploy/aws/requirements-linux-py312.txt"
cd "$release_dir"
/opt/opendart/venv/bin/python -m pip install --no-deps --no-build-isolation -e .
/opt/opendart/venv/bin/python -m pip check
/opt/opendart/venv/bin/python - <<'PY'
import cv2, importlib.metadata as m
assert cv2.__version__ == '5.0.0'
assert m.version('opencv-python') == '5.0.0.93'
assert len([d for d in m.distributions() if d.metadata['Name'].lower().startswith('opencv-')]) == 1
print('Verified single OpenCV 5.0.0.93 distribution, runtime 5.0.0')
PY
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/var/lib/opendart/matplotlib
# Preserve the full test log, and fail before serving the application if tests fail.
runuser -u opendart -- env MPLBACKEND=Agg MPLCONFIGDIR="$MPLCONFIGDIR" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /opt/opendart/venv/bin/python -m pytest -q -ra --basetemp=/var/lib/opendart/validation/pytest_tmp \
  -o cache_dir=/var/lib/opendart/validation/pytest_cache --junitxml=/var/lib/opendart/validation/pytest.xml \
  > /var/lib/opendart/validation/pytest.log 2>&1
ln -sfn "$release_dir" /opt/opendart/current
printf 'OPENDART_RESULTS=/var/lib/opendart/jobs\nOPENDART_RELEASE_SHA256=%s\n' "$release_sha" > /etc/opendart.env
chmod 644 /etc/opendart.env
install -m 644 deploy/aws/opendart.service /etc/systemd/system/opendart.service
install -m 644 deploy/aws/nginx.conf /etc/nginx/sites-available/opendart
rm -f /etc/nginx/sites-enabled/default
ln -sfn /etc/nginx/sites-available/opendart /etc/nginx/sites-enabled/opendart
cat > /etc/logrotate.d/opendart <<'EOF'
/var/log/opendart/*.log {
    daily
    rotate 3
    size 10M
    missingok
    notifempty
    copytruncate
    compress
    su opendart opendart
}
EOF
systemctl daemon-reload
systemctl enable --now opendart
nginx -t
systemctl enable --now nginx
echo 'OpenDART bootstrap complete'
