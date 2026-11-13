#!/usr/bin/env bash
# Forced command for the deploy key in ~bot/.ssh/authorized_keys:
#   command="/opt/trolobot/deploy-only.sh",no-port-forwarding,no-agent-forwarding,no-pty ssh-ed25519 AAAA...
#
# CI connects as `ssh bot@host "bash -s -- <tag>" < deploy/deploy.sh`;
# exactly that string (without the stdin content) ends up in SSH_ORIGINAL_COMMAND.
# Only the form "bash -s -- sha-<7-12 hex>" is allowed, anything else is refused.
set -euo pipefail

PATTERN='^bash -s -- sha-[0-9a-f]{7,12}$'
CMD="${SSH_ORIGINAL_COMMAND:-}"

if [[ "$CMD" =~ $PATTERN ]]; then
  # Deliberately unquoted: "bash -s -- sha-xxxxx" must be split into words so that
  # bash -s reads the script (deploy.sh) from stdin and sha-xxxxx becomes $1.
  # shellcheck disable=SC2086
  exec $CMD
fi

echo "deploy-only.sh: command not allowed: '${CMD:-<empty>}'" >&2
command -v logger >/dev/null 2>&1 && logger -t trolobot-deploy "rejected command: ${CMD:-<empty>}"
exit 1
