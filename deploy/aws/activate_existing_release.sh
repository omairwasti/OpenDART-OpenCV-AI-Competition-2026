#!/usr/bin/env bash
# Application-only release activation on the existing CloudFormation instance.
set -euo pipefail
release_dir="$1"
release_sha="$2"
[[ "$release_sha" =~ ^[a-f0-9]{64}$ ]]
[[ "$release_dir" == "/opt/opendart/releases/$release_sha" ]]
old_release=$(readlink -f /opt/opendart/current)
[[ "$old_release" == /opt/opendart/releases/* ]]
validation="/var/lib/opendart/validation/task3a6-$release_sha"
mkdir "$validation"
chown opendart:opendart "$validation"
cd "$release_dir"
/opt/opendart/venv/bin/python - <<'PY'
import hashlib,json,pathlib
root=pathlib.Path.cwd()
for name,expected in json.loads((root/'release_manifest.json').read_text()).items():
    p=root/name
    assert p.resolve().is_relative_to(root) and hashlib.sha256(p.read_bytes()).hexdigest()==expected,name
for name,expected in json.loads((root/'scientific_manifest.json').read_text()).items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
PY
# Old service remains available while the new release passes all Linux tests.
runuser -u opendart -- env PYTHONPATH="$release_dir" PYTHONDONTWRITEBYTECODE=1 \
  MPLBACKEND=Agg MPLCONFIGDIR=/var/lib/opendart/matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /opt/opendart/venv/bin/python -m pytest -q -ra \
  --basetemp="$validation/pytest_tmp" -o cache_dir="$validation/pytest_cache" \
  --junitxml="$validation/pytest.xml" > "$validation/pytest.log" 2>&1
/opt/opendart/venv/bin/python -m pip check > "$validation/pip-check.log"
# Wait for an admitted computation, and keep admission blocked during activation.
exec 9<>/var/lib/opendart/jobs/.worker.lock
flock -w 190 9
cp /etc/nginx/sites-available/opendart "$validation/nginx.before"
cp /etc/opendart.env "$validation/environment.before"
install -m 644 deploy/aws/nginx.conf /etc/nginx/sites-available/opendart
if ! nginx -t; then
  cp "$validation/nginx.before" /etc/nginx/sites-available/opendart
  exit 1
fi
ln -sfn "$release_dir" /opt/opendart/current.next
mv -Tf /opt/opendart/current.next /opt/opendart/current
sed -i "s/^OPENDART_RELEASE_SHA256=.*/OPENDART_RELEASE_SHA256=$release_sha/" /etc/opendart.env
if ! systemctl restart opendart; then
  ln -sfn "$old_release" /opt/opendart/current
  cp "$validation/environment.before" /etc/opendart.env
  cp "$validation/nginx.before" /etc/nginx/sites-available/opendart
  systemctl restart opendart
  exit 1
fi
systemctl reload nginx
ready=false
for attempt in $(seq 1 30); do
  if /opt/opendart/venv/bin/python -c 'import json,sys,urllib.request; assert json.load(urllib.request.urlopen("http://127.0.0.1:8000/healthz",timeout=2))["release_sha256"]==sys.argv[1]' "$release_sha" 2>/dev/null; then
    ready=true
    break
  fi
  sleep 1
done
if [ "$ready" != true ]; then
  ln -sfn "$old_release" /opt/opendart/current
  cp "$validation/environment.before" /etc/opendart.env
  cp "$validation/nginx.before" /etc/nginx/sites-available/opendart
  systemctl restart opendart
  systemctl reload nginx
  echo 'Health gate failed; previous application restored' >&2
  exit 1
fi
echo "Application release activated on existing instance: $release_sha"
echo "Validation: $validation"
systemctl is-active opendart nginx
