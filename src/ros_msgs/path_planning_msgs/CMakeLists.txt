cmake_minimum_required(VERSION 2.8.3)
project(path_planning_msgs)

find_package(catkin REQUIRED COMPONENTS
  common_msgs
  geometry_msgs
  std_msgs
  message_generation
  autoware_lanelet2_msgs
  geographic_msgs
  sensor_msgs
)

add_message_files(
  FILES
  VehicleState.msg
  Path.msg
  PathArray.msg
  Environment.msg
  OccupancyGrid.msg
  GridY.msg
  LinearBlur.msg
  CostmapConfig.msg
  GoalLine.msg
  Diagnostics.msg
  RolloutCostmapAndConfig.msg
  CostmapValueGrid.msg
  MPCState.msg
  MPCOutput.msg
  QuinticParam.msg
  CarlaEgoVehicleControl.msg
  CarlaEgoVehicleStatus.msg
  AckermannDrive.msg
  FrenetState.msg
  FrenetPath.msg
  PIDDebug.msg
  GlobalLink.msg
  GlobalNode.msg
  GlobalCommand.msg
  GlobalCommandList.msg
  EgoVehicleState.msg
  SystemIdentification.msg
  DestinationList.msg
  MPCStats.msg
  ReferenceLine.msg
  ReferenceSpline.msg
)

add_service_files(
  FILES
  GetGlobalLink.srv
  GetGlobalNode.srv
  GlobalRoute.srv
  GlobalStartLocation.srv
  GlobalDestinationQuery.srv
  GlobalLinkFracPoint.srv
  Lanelet2MapService.srv
  Lanelet2MapName.srv
  LaneletDestinationService.srv
  GetBehavior.srv
  RegulatoryElementService.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
  common_msgs
  autoware_lanelet2_msgs
  geographic_msgs
)

catkin_package(
  CATKIN_DEPENDS common_msgs message_runtime geometry_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
