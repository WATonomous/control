# A Node represents an junction of multiple links (ie an intersection) in the HERE SD Map logical model

Header header

# Unique ID
int32 id
# IDs of links that intersect at this node
int32[] out_link_ids
# Location of this node in some coordinate from, usually odom/map AKA UTM17N (EPSG 26917)
geometry_msgs/Point location