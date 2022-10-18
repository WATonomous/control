Header header

#commands
bool POINT
bool LINE

# if it is a point, it will connect a line to the previously drawn point
# DEPLETED TO REDUCE COMPLEXITY
# bool connect 

# if perminant defined, calling clear will not erase these points
# points will only be cleaned by calling new
bool perminant

#width of the line or point to be drawn
int8 width

#color of the object (red, green, blue etc.)
string color

#types of possible things drawable
geometry_msgs/Point pt
geometry_msgs/Point[] ln_seg 