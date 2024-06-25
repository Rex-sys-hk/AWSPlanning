# I am seeing a new goal x:45.1832 y:36.7963 t:89.6049
# I am seeing a new start x:63.5576 y:36.2744 t:39.5795

rostopic pub /move_base_simple/goal geometry_msgs/PoseStamped "header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: 'map'
pose:
    position:
        x: 63.5576
        y: 36.2744
        z: 0.0
    orientation:
        x: 0.0
        y: 0.0
        z: 0.707
        w: 0.707" -1
