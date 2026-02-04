
# 1.build
colcon build --packages-select yolov8_segment

# 2. run
source install/setup.bash

ros2 launch yolov8_segment yolov8_segment.launch.py 

 
 
# 3. echo

ros2 topic echo /segmentation/rectangles


# 4.result

header:
  stamp:
    sec: 1768382654
    nanosec: 232261888
  frame_id: class_0_score_0.82
polygon:
  points:
  - x: 531.0
    y: 394.0
    z: 0.0
  - x: 102.0
    y: 349.0
    z: 0.0
  - x: 105.0
    y: 316.0
    z: 0.0
  - x: 535.0
    y: 361.0
    z: 0.0
---

