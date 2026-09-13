#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR="${1:-/spot_logs/arthurz}"
REMOTE_USER_HOST="${2:-amrl_robot@robodata.csres.utexas.edu}"
REMOTE_DIR="${3:-/robodata/spot_logs/arthurz/foresight_bags/experiment_logs}"

# Optional: set a specific SSH private key by exporting SSH_KEY before running
# Example:
#   SSH_KEY=/home/ros/.ssh/id_ed25519 ./sync_bags.sh
SSH_OPTS=()
if [[ -n "${SSH_KEY:-}" ]]; then
  SSH_OPTS=(-i "$SSH_KEY")
fi

mkdir -p "$LOCAL_DIR"

rsync -avzP \
  -e "ssh ${SSH_OPTS[*]:-}" \
  "$LOCAL_DIR"/ \
  "${REMOTE_USER_HOST}:${REMOTE_DIR}/"
