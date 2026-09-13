#!/bin/bash
# Switch the current user's shell environment to FastDDS.
# Run this INSIDE the container after starting a fresh shell:
#     bash ~/cotnav_ws/scripts/setup_unicast_fastdds.sh
# Then `exec bash` (or exit and re-enter the container) for the changes to apply.
#
# This edits both ~/.bashrc (non-login shells) and ~/.bash_profile (login shells)
# so Cyclone settings baked into the image don't silently override FastDDS.
#
# Changes are local to the running container's writable layer; a fresh container
# created from the image will boot back to the default (Cyclone) until this
# script is run again.

set -e

update_bashrc() {
    local FILE="$1"
    [ -f "$FILE" ] || return 0

    echo "Updating $FILE"

    # Ensure RMW_IMPLEMENTATION is set to FastDDS
    if grep -q "^[[:space:]]*export RMW_IMPLEMENTATION=" "$FILE"; then
        sed -i 's|^[[:space:]]*export RMW_IMPLEMENTATION=.*|export RMW_IMPLEMENTATION=rmw_fastrtps_cpp|' "$FILE"
    else
        echo "export RMW_IMPLEMENTATION=rmw_fastrtps_cpp" >> "$FILE"
    fi

    # Comment out CycloneDDS-specific exports so they don't take effect
    sed -i 's|^[[:space:]]*export CYCLONEDDS_URI=|# export CYCLONEDDS_URI=|' "$FILE"
    sed -i 's|^[[:space:]]*export ROS_AUTOMATIC_DISCOVERY_RANGE=|# export ROS_AUTOMATIC_DISCOVERY_RANGE=|' "$FILE"

    # ROS_VERSION / ROS_DOMAIN_ID are DDS-agnostic; keep them aligned with the Cyclone script
    if grep -q "^[[:space:]]*export ROS_VERSION=" "$FILE"; then
        sed -i 's|^[[:space:]]*export ROS_VERSION=.*|export ROS_VERSION=2|' "$FILE"
    else
        echo "export ROS_VERSION=2" >> "$FILE"
    fi
}

update_bashrc "$HOME/.bashrc"
update_bashrc "$HOME/.bash_profile"

echo
echo "Done. Run 'exec bash' or open a new shell, then verify with:"
echo "    echo \$RMW_IMPLEMENTATION   # expect rmw_fastrtps_cpp"
echo "    echo \$CYCLONEDDS_URI       # expect empty"
