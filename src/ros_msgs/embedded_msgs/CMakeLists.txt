cmake_minimum_required(VERSION 2.8.3)
project(embedded_msgs)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  message_generation
)

add_message_files(
  FILES
  DesiredOutput.msg
  AutoModeFeedback.msg
  Braking_CAN.msg
  IMUData_CAN.msg
  Speed_CAN.msg
  SteeringAngle_CAN.msg
  Torque_CAN.msg
  WheelEncoder_CAN.msg
  Transmission_State_CAN.msg
  Trans_Req_CAN.msg
  Steering_State_CAN.msg
  LowSpeedBus_ControlRequest.msg
  LockLightingRequest.msg
)

generate_messages(
   DEPENDENCIES
   std_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES embedded_msgs
  CATKIN_DEPENDS message_runtime
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

