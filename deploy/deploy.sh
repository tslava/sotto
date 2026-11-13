#!/usr/bin/env bash
# Runs on the host: it arrives over SSH via `bash -s -- <tag>` (see
# deploy/deploy-only.sh and .github/workflows/deploy.yml); the script is never
# stored on the host as a separate file, so the host always runs the current
# version from the repository.
#
# Writes IMAGE_TAG/IMAGE_TAG_PREV to deploy.env, brings docker compose up,
# waits for healthy and rolls back to the previous tag if the container did not come up.
set -euo pipefail

APP_DIR="/opt/trolobot"
ENV_FILE="deploy.env"
SERVICE="bot"
HEALTH_TIMEOUT_SEC=60

usage() {
  echo "usage: deploy.sh <image-tag>  (for example: sha-abc1234)" >&2
  exit 1
}

TAG="${1:-}"
[ -n "$TAG" ] || usage

cd "$APP_DIR"

if [ ! -f "$ENV_FILE" ]; then
  echo "deploy.sh: $APP_DIR/$ENV_FILE not found; run deploy/setup-host.sh first" >&2
  exit 1
fi

# The current tag before the overwrite: we roll back to it on failure. Empty means
# the first deploy (deploy.env has no IMAGE_TAG= line yet), so there is nothing to roll back to.
PREV_TAG="$(grep -E '^IMAGE_TAG=' "$ENV_FILE" | tail -n1 | cut -d= -f2- || true)"
GHCR_OWNER_LINE="$(grep -E '^GHCR_OWNER=' "$ENV_FILE" || echo 'GHCR_OWNER=tslava')"

write_deploy_env() {
  # $1 = IMAGE_TAG, $2 = IMAGE_TAG_PREV
  local tmp
  tmp="$(mktemp "${ENV_FILE}.XXXXXX")"
  {
    echo "$GHCR_OWNER_LINE"
    echo "IMAGE_TAG=$1"
    echo "IMAGE_TAG_PREV=$2"
  } >"$tmp"
  mv "$tmp" "$ENV_FILE"
}

# Common rollback point: called when pull/up -d fails and when the healthcheck
# times out, so IMAGE_TAG in deploy.env never stays on a failed tag
# whichever way the deploy fails.
rollback() {
  # $1 = reason (for the log)
  echo "deploy.sh: $1; rolling back to ${PREV_TAG:-<none>}" >&2

  if [ -z "$PREV_TAG" ]; then
    echo "deploy.sh: no previous IMAGE_TAG (first deploy); nothing to roll back to" >&2
    exit 1
  fi

  # Take the logs BEFORE recreating: compose up -d removes the failed container,
  # and docker logs would then say "No such container".
  local failed_cid
  failed_cid="$(docker compose --env-file "$ENV_FILE" ps -aq "$SERVICE" | head -1 || true)"
  if [ -n "$failed_cid" ]; then
    echo "deploy.sh: last 50 log lines of the failed container ($TAG):" >&2
    docker logs --tail 50 "$failed_cid" >&2 || true
  fi

  write_deploy_env "$PREV_TAG" "$PREV_TAG"
  if ! docker compose --env-file "$ENV_FILE" up -d; then
    echo "deploy.sh: the rollback to $PREV_TAG did not come up either (docker compose up -d failed)" >&2
  fi

  exit 1
}

wait_healthy() {
  local cid elapsed=0 status
  cid="$(docker compose --env-file "$ENV_FILE" ps -q "$SERVICE" || true)"
  if [ -z "$cid" ]; then
    return 1
  fi
  while [ "$elapsed" -lt "$HEALTH_TIMEOUT_SEC" ]; do
    status="$(docker inspect --format '{{.State.Health.Status}}' "$cid" 2>/dev/null || echo "unknown")"
    if [ "$status" = "healthy" ]; then
      return 0
    fi
    if [ "$status" = "unhealthy" ]; then
      # Do not wait out the remaining timeout if docker has already decided the container is bad.
      return 1
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done
  return 1
}

echo "deploy.sh: deploying $TAG (previous tag: ${PREV_TAG:-<none>})"
write_deploy_env "$TAG" "$PREV_TAG"

if ! docker compose --env-file "$ENV_FILE" pull; then
  rollback "docker compose pull $TAG failed"
fi

if ! docker compose --env-file "$ENV_FILE" up -d; then
  rollback "docker compose up -d for $TAG failed"
fi

if wait_healthy; then
  echo "deploy.sh: $TAG healthy, done"
  exit 0
fi

rollback "$TAG did not become healthy within ${HEALTH_TIMEOUT_SEC}s"
