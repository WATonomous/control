cmake_minimum_required(VERSION 2.8.3)
project(perception_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  # other packages containing messages
  geometry_msgs
  std_msgs
  common_msgs
  sensor_msgs
)

add_message_files(
  FILES
  RoadLinesClassMask.msg
  TrafficLightBoxes.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
  common_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS
  message_runtime
  geometry_msgs
  std_msgs
  common_msgs
  sensor_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
