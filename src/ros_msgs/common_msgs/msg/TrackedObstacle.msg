# Message depicting the bounding box of a tracked obstacle, including it's predictions and observation history
# Tracked objects will be published in the odom frame

Header header

# The bounding box and current position of the obstacle in question
common_msgs/Obstacle obstacle

# Note the velocity in the observation history and predicted states is the Kalman Filter estimate, 
# which is NOT accuate. The obstacle above contains a simpler more accurate estimate. 

# History of Observations (duration is configurable) The last element in observation_history corresponds to the current state
common_msgs/TrackedObstacleState[] observation_history
# Linear future predictions based on velocity and position
common_msgs/TrackedObstacleState[] predicted_states