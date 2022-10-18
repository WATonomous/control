# This message type represents the geometric and traffic attributes
# of a lane from some HD Map.
Header header

# Id used to categorize stitched lanes by starting lane
int32 lane_id

# Polyline representing center line trajectory of this lane.
# Direction of travel is implied by the ordering of points.
# the trajectory is after stitching.
geometry_msgs/Point[] trajectory

# Light ROS msg wrapper around a polyline to represent left and right
# boundaries of this lane.
# The ordering of points should correspond to trajectory.
HDMLaneLine[] left_boundaries
HDMLaneLine[] right_boundaries


# Additional HERE HDM specific context.
# HERE Topology Geometry Link IDs.
uint32[] link_ids
