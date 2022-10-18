Header header

# This desired theta is measured in degrees, with 0 being perfectly straight, and anti-clockwise (turning "left") being positive.
float32 theta

# Newton meters
float32 torque

# If acceleration is preferred instead of the estimated torque
# m/s^2
float32 acceleration

# Two states: 0 = stopped, 1 = driving
int8 state
