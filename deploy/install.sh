#!/usr/bin/env bash
# Front Page installer / updater for Raspberry Pi OS (64-bit). Idempotent:
# run it again after every `git pull`.
#
#   ./deploy/install.sh
#
# It creates a venv, installs requirements, migrates the database, installs
# and (re)starts the systemd units, and kicks off a fetch in the background.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_USER="$(id -un)"
PORT=9092
cd "$REPO"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

if [ "$RUN_USER" = "root" ]; then
  echo "Run this as your normal user (it uses sudo only for systemd), not as root." >&2
  exit 1
fi
case "$REPO" in
  *[[:space:]]*) echo "The repo path contains spaces, which systemd units cannot handle: $REPO" >&2; exit 1 ;;
esac

say "Checking Python"
PYTHON="${PYTHON:-python3}"
if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
  echo "Python 3.11 or newer is required (found: $("$PYTHON" --version 2>&1))." >&2
  exit 1
fi
if ! "$PYTHON" -c 'import venv, ensurepip' 2>/dev/null; then
  say "Installing python3-venv"
  sudo apt-get update -qq && sudo apt-get install -y python3-venv
fi

say "Creating virtualenv and installing requirements"
[ -x .venv/bin/python ] || "$PYTHON" -m venv .venv
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r requirements.txt

say "Preparing data directory and database"
mkdir -p data/images data/models
.venv/bin/python -m core.db

say "Installing systemd units"
for unit in frontpage.service frontpage-fetch.service frontpage-fetch.timer; do
  sed -e "s|@@REPO@@|$REPO|g" -e "s|@@USER@@|$RUN_USER|g" "deploy/$unit" \
    | sudo tee "/etc/systemd/system/$unit" >/dev/null
done
sudo systemctl daemon-reload
sudo systemctl enable --quiet frontpage.service frontpage-fetch.timer
sudo systemctl restart frontpage.service
sudo systemctl restart frontpage-fetch.timer

say "Starting a fetch in the background"
# The first run downloads the embedding model (~90 MB) and can take a few
# minutes; it does not block this script.
sudo systemctl start --no-block frontpage-fetch.service

say "Waiting for the web app"
ok=0
for _ in $(seq 1 20); do
  if .venv/bin/python - "$PORT" <<'PY' 2>/dev/null
import sys, urllib.request
urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/healthz", timeout=2)
PY
  then ok=1; break; fi
  sleep 1
done
if [ "$ok" = 1 ]; then
  echo "Front Page is running on http://127.0.0.1:$PORT"
else
  echo "The web app did not answer yet. Check: journalctl -u frontpage -n 50" >&2
fi

cat <<MSG

Done.

  Watch the fetch:   journalctl -u frontpage-fetch -f
  Web app logs:      journalctl -u frontpage -f
  Check the feeds:   .venv/bin/python -m core.fetch --check

Expose it on your tailnet (pick one):

  # on its own HTTPS port, next to your other apps:
  sudo tailscale serve --bg --https=$PORT http://127.0.0.1:$PORT
  #   -> https://<pi-name>.<tailnet>.ts.net:$PORT

  # or as the default https://<pi-name>.<tailnet>.ts.net/ :
  sudo tailscale serve --bg $PORT

MSG
