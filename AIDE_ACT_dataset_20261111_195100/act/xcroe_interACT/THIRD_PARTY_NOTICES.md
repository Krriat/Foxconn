# Third-party notices

The segmented encoder and synchronized multi-arm decoder design in
`src/xcore_interact/policy/modeling_interact3d.py` is derived from the
InterACT architecture:

- Repository: https://github.com/Soltanilara/InterACT-LeRobot
- Paper: https://arxiv.org/abs/2409.07914
- Audited upstream commit: `45ad7c69c010721e27b2ada08cf99481f9c7809c`
- Upstream license: Apache License 2.0

The implementation here is rewritten for LeRobot 0.4.x, a fixed 16-D xCore
bimanual schema, three RGB cameras, and a head-camera XYZRGB PointNet branch.
The upstream project itself uses RGB and does not provide this 3D branch.

`native/pico_dual_recording.cpp` is based on the existing local xCore dual-PICO
controller and adds ROS dataset telemetry plus optional gripper interfaces.

