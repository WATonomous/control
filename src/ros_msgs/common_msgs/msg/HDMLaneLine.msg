# This message type represents the geometric and traffic attributes
# of a lane line from some HD Map.
Header header

# Unique ID for this lane line, which is an OSM Way concept.
int32 lane_line_id

geometry_msgs/Point[] polyline

string COLOR_UNKNOWN=UNKNOWN
string COLOR_WHITE=WHITE
string COLOR_YELLOW=YELLOW
string color

string PATTERN_UNKNOWN=UNKNOWN
string PATTERN_SOLID=SOLID
string PATTERN_DASHED=DASHED
string pattern

