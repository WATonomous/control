cmake_minimum_required(VERSION 2.8.3)
project(wato_topics)

# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_FLAGS "-Wall")

find_package(catkin)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE include/)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
)

add_message_files(
  FILES
  path_planning_topics.msg
  perception_topics.msg
  processing_topics.msg
  sensor_topics.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
  std_msgs
  INCLUDE_DIRS include/
)
