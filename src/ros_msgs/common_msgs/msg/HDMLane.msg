# This message type represents the geometric and traffic attributes
# of a lane from some HD Map.
Header header

# Unique ID for this lane.
int32 NULL_LANE_ID = 0
int32 lane_id

# Polyline representing center line trajectory of this lane.
# Direction of travel is implied by the ordering of points.
geometry_msgs/Point[] trajectory

# Light ROS msg wrapper around a polyline to represent left and right
# boundaries of this lane.
# The ordering of points should correspond to trajectory.
HDMLaneLine left_boundary
HDMLaneLine right_boundary

# List of lane IDs. These lanes are upcoming connections from this lane.
int32[] next

# Lane ID of the lane on the left/right of this lane.
# Left and right are from the perspective of an observer
# looking towards the next trajectory point.
int32 left
int32 right

# Additional HERE HDM specific context.
# HERE Topology Geometry Link IDs.
int64 start_link_id
int64 end_link_id

