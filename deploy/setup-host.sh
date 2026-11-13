#!/usr/bin/env bash
# First-time setup of the VPS for trolobot. Idempotent: safe to re-run.
# Run once by the owner under sudo on the host itself:
#
#   sudo bash deploy/setup-host.sh \
#     --pubkey-file /path/to/deploy_key.pub \
#     [--compose-path /path/to/docker-compose.yml] \
#     [--repo tslava/trolobot] [--ref main]
#
# Without --compose-path, docker-compose.yml is downloaded from GitHub raw at branch --ref.
# Without --pubkey-file, authorized_keys is left alone for the deploy key: add the
# line by hand (the script prints a ready template).
set -euo pipefail

APP_DIR="/opt/trolobot"
BOT_USER="bot"
DATA_UID=10001
DATA_GID=10001
REPO="tslava/trolobot"
REF="main"
COMPOSE_PATH=""
PUBKEY_FILE=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { echo "[setup-host] $*"; }
warn() { echo "[setup-host] WARNING: $*" >&2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --compose-path) COMPOSE_PATH="$2"; shift 2 ;;
    --pubkey-file) PUBKEY_FILE="$2"; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    --ref) REF="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [ "$(id -u)" -ne 0 ]; then
  echo "run under sudo/root" >&2
  exit 1
fi

# ---- 1. Docker + compose plugin ---------------------------------------------
if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  log "docker is already installed, skipping"
else
  log "installing docker from the official repository"
  apt-get update -y
  apt-get install -y ca-certificates curl gnupg
  install -m 0755 -d /etc/apt/keyrings
  if [ ! -f /etc/apt/keyrings/docker.asc ]; then
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    chmod a+r /etc/apt/keyrings/docker.asc
  fi
  . /etc/os-release
  CODENAME="${VERSION_CODENAME:-$(lsb_release -cs 2>/dev/null || echo noble)}"
  cat >/etc/apt/sources.list.d/docker.list <<EOF
deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${CODENAME} stable
EOF
  apt-get update -y
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

# ---- 2. User bot ------------------------------------------------------------
if id -u "$BOT_USER" >/dev/null 2>&1; then
  log "user $BOT_USER already exists, skipping"
else
  log "creating user $BOT_USER (no password, home directory)"
  useradd --create-home --shell /bin/bash "$BOT_USER"
  passwd -l "$BOT_USER" >/dev/null   # lock the password: SSH key only
fi
usermod -aG docker "$BOT_USER"

# ---- 3. /opt/trolobot and data -------------------------------------------------
mkdir -p "$APP_DIR"
chown "$BOT_USER:$BOT_USER" "$APP_DIR"
chmod 750 "$APP_DIR"

mkdir -p "$APP_DIR/data"
chown "$DATA_UID:$DATA_GID" "$APP_DIR/data"
chmod 750 "$APP_DIR/data"

mkdir -p "$APP_DIR/backups"
chown "$BOT_USER:$BOT_USER" "$APP_DIR/backups"
chmod 750 "$APP_DIR/backups"

# ---- 4. docker-compose.yml ----------------------------------------------------
if [ -f "$APP_DIR/docker-compose.yml" ]; then
  log "$APP_DIR/docker-compose.yml already exists, not overwriting (delete the file to refresh it from the source)"
elif [ -n "$COMPOSE_PATH" ]; then
  log "copying docker-compose.yml from $COMPOSE_PATH"
  cp "$COMPOSE_PATH" "$APP_DIR/docker-compose.yml"
else
  log "downloading docker-compose.yml from github.com/${REPO}@${REF}"
  curl -fsSL "https://raw.githubusercontent.com/${REPO}/${REF}/docker-compose.yml" \
    -o "$APP_DIR/docker-compose.yml"
fi
chown "$BOT_USER:$BOT_USER" "$APP_DIR/docker-compose.yml"

# ---- 5. deploy.sh / deploy-only.sh --------------------------------------------
# deploy.sh is not stored on the host: CI pipes it through stdin every time
# (`ssh ... "bash -s -- <tag>" < deploy/deploy.sh`), so the host always runs
# the current version from the repository. The host only needs the
# forced-command wrapper deploy-only.sh.
if [ -f "$SCRIPT_DIR/deploy-only.sh" ]; then
  cp "$SCRIPT_DIR/deploy-only.sh" "$APP_DIR/deploy-only.sh"
else
  log "downloading deploy-only.sh from github.com/${REPO}@${REF}"
  curl -fsSL "https://raw.githubusercontent.com/${REPO}/${REF}/deploy/deploy-only.sh" \
    -o "$APP_DIR/deploy-only.sh"
fi
chown "$BOT_USER:$BOT_USER" "$APP_DIR/deploy-only.sh"
chmod 750 "$APP_DIR/deploy-only.sh"

# ---- 6. .env (application secrets) ---------------------------------------------
if [ -f "$APP_DIR/.env" ]; then
  log "$APP_DIR/.env already exists, leaving it alone"
else
  ENV_EXAMPLE=""
  if [ -f "$SCRIPT_DIR/../.env.example" ]; then
    ENV_EXAMPLE="$SCRIPT_DIR/../.env.example"
  fi
  if [ -n "$ENV_EXAMPLE" ]; then
    cp "$ENV_EXAMPLE" "$APP_DIR/.env"
  else
    curl -fsSL "https://raw.githubusercontent.com/${REPO}/${REF}/.env.example" \
      -o "$APP_DIR/.env"
  fi
  chown "$BOT_USER:$BOT_USER" "$APP_DIR/.env"
  chmod 600 "$APP_DIR/.env"
  warn "$APP_DIR/.env created empty from .env.example: FILL IT IN by hand (BOT_TOKEN, OPENROUTER_API_KEY, ...) before the first start"
fi

# ---- 7. deploy.env (IMAGE_TAG / GHCR_OWNER, not a secret) ----------------------
if [ -f "$APP_DIR/deploy.env" ]; then
  log "$APP_DIR/deploy.env already exists, leaving it alone"
else
  GHCR_OWNER="$(echo "$REPO" | cut -d/ -f1)"
  cat >"$APP_DIR/deploy.env" <<EOF
GHCR_OWNER=${GHCR_OWNER}
IMAGE_TAG=latest
EOF
  chown "$BOT_USER:$BOT_USER" "$APP_DIR/deploy.env"
  chmod 644 "$APP_DIR/deploy.env"
fi

# ---- 8. Deploy SSH key (authorized_keys with a forced command) ---------------
BOT_HOME="$(getent passwd "$BOT_USER" | cut -d: -f6)"
SSH_DIR="$BOT_HOME/.ssh"
AUTH_KEYS="$SSH_DIR/authorized_keys"
mkdir -p "$SSH_DIR"
chmod 700 "$SSH_DIR"
touch "$AUTH_KEYS"
chmod 600 "$AUTH_KEYS"

if [ -n "$PUBKEY_FILE" ]; then
  PUBKEY="$(cat "$PUBKEY_FILE")"
  RESTRICTED_LINE="command=\"$APP_DIR/deploy-only.sh\",no-port-forwarding,no-agent-forwarding,no-pty ${PUBKEY}"
  if grep -qF "${PUBKEY#* }" "$AUTH_KEYS" 2>/dev/null; then
    log "the public key is already in $AUTH_KEYS, skipping"
  else
    echo "$RESTRICTED_LINE" >>"$AUTH_KEYS"
    log "added the restricted deploy key to $AUTH_KEYS"
  fi
else
  warn "--pubkey-file not given: add this line to $AUTH_KEYS by hand:"
  echo "  command=\"$APP_DIR/deploy-only.sh\",no-port-forwarding,no-agent-forwarding,no-pty <contents of the deploy key .pub>"
fi
chown -R "$BOT_USER:$BOT_USER" "$SSH_DIR"

# ---- 9. ufw: only SSH from outside ----------------------------------------------
if ! command -v ufw >/dev/null 2>&1; then
  apt-get install -y ufw
fi
if ufw status | grep -q "^Status: active"; then
  # The firewall is already set up by the owner (the host may run other services,
  # for example a gateway on 443): leave the rules alone, only make sure SSH is allowed.
  ufw allow OpenSSH >/dev/null
  log "ufw is already active: existing rules kept, SSH allowed:"
  ufw status numbered | sed 's/^/    /'
else
  ufw allow OpenSSH >/dev/null
  ufw default deny incoming >/dev/null
  ufw --force enable >/dev/null
  log "ufw: allow OpenSSH, default deny incoming, enabled"
fi

# ---- 10. sshd: check only, no changes (do not break the owner's access) ------
SSHD_EFFECTIVE="$(sshd -T 2>/dev/null || true)"
PASS_AUTH="$(echo "$SSHD_EFFECTIVE" | awk '/^passwordauthentication/{print $2}')"
ROOT_LOGIN="$(echo "$SSHD_EFFECTIVE" | awk '/^permitrootlogin/{print $2}')"
if [ "$PASS_AUTH" = "yes" ]; then
  warn "sshd: PasswordAuthentication is still yes. If you (the owner) log in with a password," \
       "set up key access for yourself first, then set 'PasswordAuthentication no' in /etc/ssh/sshd_config" \
       "and restart sshd; this script does NOT do it, so it cannot cut off your access."
fi
if [ "$ROOT_LOGIN" != "prohibit-password" ] && [ "$ROOT_LOGIN" != "no" ]; then
  warn "sshd: PermitRootLogin=${ROOT_LOGIN:-unknown}. Recommended: 'PermitRootLogin prohibit-password' in /etc/ssh/sshd_config."
fi

# ---- 11. docker login to ghcr.io (for a private package) ---------------------
log "if the package ghcr.io/${REPO} is private, log user $BOT_USER in to ghcr.io:"
echo "  sudo -u $BOT_USER docker login ghcr.io -u <github-username> --password-stdin <<< \"<PAT with read:packages>\""
echo "  (or make the package public in the GHCR settings; then no login is needed)"

# ---- 12. weekly backup of bot.db -----------------------------------------------
if ! command -v sqlite3 >/dev/null 2>&1; then
  apt-get install -y sqlite3
fi

BACKUP_SCRIPT="$APP_DIR/backup.sh"
cat >"$BACKUP_SCRIPT" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd /opt/trolobot
sqlite3 data/bot.db ".backup backups/bot-$(date +%F).db"
# keep the last 8 copies
ls -1t backups/bot-*.db 2>/dev/null | tail -n +9 | xargs -r rm -f
EOF
chown "$BOT_USER:$BOT_USER" "$BACKUP_SCRIPT"
chmod 750 "$BACKUP_SCRIPT"

CRON_MARKER="# trolobot weekly backup"
CRON_LINE="0 3 * * 0 $BACKUP_SCRIPT >>/opt/trolobot/backups/backup.log 2>&1 $CRON_MARKER"
EXISTING_CRON="$(crontab -u "$BOT_USER" -l 2>/dev/null || true)"
if echo "$EXISTING_CRON" | grep -qF "$CRON_MARKER"; then
  log "cron backup already set up, skipping"
else
  { echo "$EXISTING_CRON"; echo "$CRON_LINE"; } | grep -v '^$' | crontab -u "$BOT_USER" -
  log "added a weekly cron backup (Sun 03:00) as $BOT_USER"
fi

log "done. Next, by hand: fill in $APP_DIR/.env, docker login if the package is private, then push to main."
