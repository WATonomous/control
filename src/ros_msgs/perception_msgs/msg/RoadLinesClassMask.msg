Header header

# Position and orientation of the vehicle.
geometry_msgs/Pose pose

# Pixel-wise classification of image for roadlines. This should not be in BEV form yet.
# This array is a 2D class matrix with same (width, height) size as the input image
# to the roadline semantic segmentation network.
# ex.
# [0 0 0 1 1 2 2 2 2]
# [0 0 1 1 1 2 2 2 2]
# [0 1 1 1 1 1 1 1 1]
sensor_msgs/Image[] masks

# Each index of this array maps integer values in the mask 2D array to a class name.
# ex. ['Void','Markings-general','Markings-crosswalk']
string[] classes
