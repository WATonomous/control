Header header

# Parameters
float64 Kp
float64 Ki
float64 Kd

float64 sp # setpoint
float64 pv # process variable (output of plant)
float64 err # sp - pv
float64 u # actuation (output of controller, input to plant)

# Components
float64 p
float64 i
float64 d

# Feedforward term (optional)
float64 ff