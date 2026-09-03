#!/usr/bin/env bash
set -uo pipefail

IFACE="${1:-eth0}"
OUTDIR="${2:-/tmp/payload_net_logs}"
INTERVAL="${INTERVAL:-1}"

mkdir -p "$OUTDIR"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$OUTDIR/run_$TS"
mkdir -p "$RUN_DIR"

echo "Logging interface: $IFACE"
echo "Output dir: $RUN_DIR"

cleanup() {
  echo
  echo "Stopping loggers..."
  jobs -p | xargs -r kill 2>/dev/null || true
  wait || true
  echo "Done. Logs saved to: $RUN_DIR"
}
trap cleanup EXIT INT TERM

# Basic system snapshot
{
  echo "===== START $(date) ====="
  echo
  echo "===== uname -a ====="
  uname -a
  echo
  echo "===== ip -br link ====="
  ip -br link
  echo
  echo "===== ip -br addr ====="
  ip -br addr
  echo
  echo "===== ip route ====="
  ip route
  echo
  echo "===== ip rule ====="
  ip rule
  echo
  echo "===== nmcli connection show ====="
  nmcli connection show || true
  echo
  echo "===== nmcli device status ====="
  nmcli device status || true
  echo
  echo "===== ethtool $IFACE ====="
  ethtool "$IFACE" || true
  echo
  echo "===== ethtool -S $IFACE ====="
  ethtool -S "$IFACE" || true
  echo
  echo "===== ip -s link show $IFACE ====="
  ip -s link show "$IFACE" || true
  echo
  echo "===== END SNAPSHOT ====="
} > "$RUN_DIR/start_snapshot.txt" 2>&1

# Kernel logs
stdbuf -oL journalctl -k -f \
  > "$RUN_DIR/journal_kernel.log" 2>&1 &
P1=$!

# NetworkManager logs
stdbuf -oL journalctl -u NetworkManager -f \
  > "$RUN_DIR/journal_networkmanager.log" 2>&1 &
P2=$!

# Link change monitor
stdbuf -oL ip monitor link dev "$IFACE" \
  > "$RUN_DIR/ip_monitor_link.log" 2>&1 &
P3=$!

# Route changes
stdbuf -oL ip monitor route \
  > "$RUN_DIR/ip_monitor_route.log" 2>&1 &
P4=$!

# Neighbor changes
stdbuf -oL ip monitor neigh dev "$IFACE" \
  > "$RUN_DIR/ip_monitor_neigh.log" 2>&1 &
P5=$!

# Continuous polling snapshot
(
  while true; do
    {
      echo "===== $(date '+%F %T') ====="
      echo "--- ip -br link show $IFACE ---"
      ip -br link show "$IFACE" || true
      echo
      echo "--- ethtool $IFACE ---"
      ethtool "$IFACE" | egrep 'Speed:|Duplex:|Link detected:' || true
      echo
      echo "--- ip -s link show $IFACE ---"
      ip -s link show "$IFACE" || true
      echo
      echo "--- ethtool -S $IFACE (filtered) ---"
      ethtool -S "$IFACE" 2>/dev/null | egrep -i 'drop|err|miss|fifo|buffer|timeout|carrier|reset' || true
      echo
      echo "--- ip route ---"
      ip route || true
      echo
      echo "--- ip neigh show dev $IFACE ---"
      ip neigh show dev "$IFACE" || true
      echo
      echo "--- nmcli device status ---"
      nmcli device status || true
      echo
      echo "--- nmcli -f GENERAL.STATE,GENERAL.CONNECTION,IP4.GATEWAY,IP4.ADDRESS device show $IFACE ---"
      nmcli -f GENERAL.STATE,GENERAL.CONNECTION,IP4.GATEWAY,IP4.ADDRESS device show "$IFACE" || true
      echo
    } >> "$RUN_DIR/polling.log" 2>&1
    sleep "$INTERVAL"
  done
) &
P6=$!

# Optional ping to Spot payload gateway
if ip route | grep -q '192\.168\.50\.'; then
  stdbuf -oL ping -I "$IFACE" 192.168.50.3 \
    > "$RUN_DIR/ping_spot_gateway.log" 2>&1 &
  P7=$!
fi

# Optional ping to public IP
stdbuf -oL ping 8.8.8.8 \
  > "$RUN_DIR/ping_public.log" 2>&1 &
P8=$!

echo
echo "Loggers running."
echo "Reproduce the failure now."
echo "Press Ctrl-C when done."
echo

wait
