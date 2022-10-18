# A Link is part of the HERE SD Map Logical Model
# It represents a way to traverse ref_node_id -> nref_node_id
# OR nref_node_id -> ref_node_id, implying that all Links are bi-directional
# In the future, when we can no longer assume that all Linds are bi-directional,
# this logical model will be updated do include directionality information

Header header

# Unique ID of the link
int32 id
# Aribitrarily decided reference node, sets the arbitrary direction of the link
int32 ref_node_id
# The node at the opposite side as ref_node
int32 nref_node_id
# Length of the link, assumed to be in meters
float64 length
# Represents what kind of road the Link is (residential vs highway), currently not used in routing
int32 functional_class
# Also not used in routing
string street_name