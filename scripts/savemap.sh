ros2 run nav2_map_server map_saver_cli \
  -f $HOME/nav_ws/src/spot_nav/spot_nav2/spot_nav/maps/alumni_center_map \
  --ros-args \
    -p save_map_timeout:=10000.0 \
    -p free_thresh:=0.25 \
    -p occupied_thresh:=0.65
