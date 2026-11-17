# Timestamp and configured external-reference frame.
std_msgs/Header header
# TCP position in metres and orientation as a unit quaternion.
geometry_msgs/Pose pose
# The same TCP orientation as SDK XYZ Euler [roll, pitch, yaw], radians.
float64[3] orientation_rpy
