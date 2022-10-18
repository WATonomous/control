Header header
common_msgs/StopLine[] stop_lines

# Each common_msgs/StopLine has a geometry_msgs/Point 
# specifying the location of the traffic regulation feature that it belongs to. 
# This msg is used for RViz debugging, not useful otherwise.
geometry_msgs/Point[] obj_points