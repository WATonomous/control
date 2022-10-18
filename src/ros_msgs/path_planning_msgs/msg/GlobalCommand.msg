Header header

# UTM easting, northing
float64 x
float64 y

# Instruction enums
int8 NONE = 0
int8 LEFT_TURN = 1
int8 RIGHT_TURN = 2
int8 STRAIGHT = 3
int8 STOP = 4

int8 instruction

int64[] link_id
