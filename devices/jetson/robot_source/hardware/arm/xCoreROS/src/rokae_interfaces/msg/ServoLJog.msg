# Absolute TCP target in the configured external-reference frame.
# Translation is metres; orientation is a unit quaternion.
geometry_msgs/Pose pose
# Normalized elbow input in [-1, 1] (PICO horizontal X; legacy field name).
# The driver applies the deadband and integrates
# the elbow at a fixed 5 degrees/s outside the deadband while commands are fresh.
float64 joystick_y
