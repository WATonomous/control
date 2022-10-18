Header header

# Frenet planner input parameters

# current state of the vehicle
FrenetState initial

geometry_msgs/Point[] reference_trajectory

# planning horizon (plan from time 0 to time T)
float64 T

# time step
float64 DT

float64 lateral_deviation

# target longitudinal speed
float64 target_speed

# parameters used to generate controller setpoints
float64 lookahead_dist
float64 steering_lookahead_dist


# Frenet planner output

float64[] s # discretized longitudinal polynomial
float64[] d # discretized lateral polynomial

# Costs
float64 cd
float64 cv
float64 cf

float64 max_speed
float64 max_accel
float64 max_curvature
bool has_collision


# The frenet planner outputs lateral deviation and longitudinal speed/acceleration/jerk
#   setpoints for downstream controllers.
float64 controller_lateral_deviation
float64 controller_yaw # target heading (rad)
float64 controller_yaw_d # target change in heading (rad/s)
float64 controller_speed # target speed (m/s)
float64 controller_acceleration # target acceleration (m/s^2)
float64 controller_jerk # target jerk (m/s^3)
