#!/bin/bash
# ROS2 Bag Play Script
# Usage: ./bag_play.sh <path_to_bag> [additional ros2 bag play args]
#
# Comment/uncomment topics below to select which to play back.
# All uncommented topics will be passed via --topics to ros2 bag play.

BAG_PATH="${1}"
shift

if [ -z "$BAG_PATH" ]; then
    echo "Usage: $0 <path_to_bag> [additional ros2 bag play args]"
    exit 1
fi

# ── Topics ────────────────────────────────────────────────────────────────────
# Comment out any topics you do NOT want to play back.

TOPICS=(
    # ── Velodyne lidar ───────────────────────────────────────────────────────
    /velodyne_points
    /velodyne_packets

    # ── VectorNav IMU / GNSS ─────────────────────────────────────────────────
    /vectornav/imu
    /vectornav/imu_uncompensated
    /vectornav/gnss
    /vectornav/pose
    /vectornav/velocity_body
    /vectornav/magnetic
    /vectornav/pressure
    /vectornav/temperature

    # ── VectorNav raw groups ─────────────────────────────────────────────────
    # /vectornav/raw/common
    # /vectornav/raw/ins
    # /vectornav/raw/gps
    # /vectornav/raw/gps2
    # /vectornav/raw/imu
    # /vectornav/raw/time
    # /vectornav/raw/attitude

    # ── VectorNav time references ────────────────────────────────────────────
    # /vectornav/time_gps
    # /vectornav/time_pps
    # /vectornav/time_syncin
    # /vectornav/time_startup

    # ── TF / diagnostics / misc ──────────────────────────────────────────────
    # /tf_static
    # /diagnostics
    # /rosout
    # /parameter_events
    # /events/write_split
)
# ── End Topics ────────────────────────────────────────────────────────────────

if [ ${#TOPICS[@]} -eq 0 ]; then
    echo "No topics selected. Uncomment topics in $0 to play them back."
    exit 1
fi

echo "Playing bag: $BAG_PATH"
echo "Topics (${#TOPICS[@]}):"
for t in "${TOPICS[@]}"; do
    echo "  $t"
done
echo ""

ros2 bag play "$BAG_PATH" --topics "${TOPICS[@]}" "$@"
