cmake_minimum_required(VERSION 2.8.3)
project(common_msgs)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  std_msgs
	sensor_msgs
  message_generation
)

add_message_files(
  FILES
  TrafficLight.msg
  TrafficLightList.msg
  TrafficSign.msg
  TrafficSignList.msg
  LaneLine.msg
	SemanticMapElement.msg
  SemanticMapEnv.msg
  StopLine.msg
  StopLineList.msg
  ParkingSpot.msg
  Obstacle.msg
  ObstacleList.msg
  RoadLinesList.msg
  IntArray.msg
  Int2dGrid.msg
  Float64Array.msg
  Obstacle.msg
  HDMLaneLine.msg
  HDMLane.msg
  HDMLaneList.msg
  HDMStitchedLane.msg
  HDMStitchedLaneList.msg
  TrackedObstacle.msg
  TrackedObstacleList.msg
  TrackedObstacleState.msg
)

add_service_files(
  FILES
  EgoLocalizationMap.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
  common_msgs
	sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
