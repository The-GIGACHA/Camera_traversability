cmake_minimum_required(VERSION 3.0.2)
project(campus_delivery_auth)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  message_filters
)

# src/ 아래 Python 라이브러리(campus_delivery_auth) 설치
catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS rospy std_msgs sensor_msgs cv_bridge message_filters
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# 실행 노드 설치 (rosrun/roslaunch 대상)
catkin_install_python(PROGRAMS
  scripts/vision_auth_node.py
  scripts/cargo_actuator_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
