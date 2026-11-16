#!/usr/bin/env python3
"""Publish a manually configured test target in geometry_msgs/msg/PoseStamped format."""
import argparse
import json
import math

# 在这里填写已经确认的机械臂坐标，单位米。
# 必须填写三个数，None用于防止忘记修改就发布一个默认运动目标。
XYZ = [0.548118,0.155080,-0.154184]
TOPIC = '/vision/material_coord'
FRAME_ID = 'base_link'  # 必须匹配conf中的frame；本脚本不转换坐标
RATE_HZ = 2.0
CLASS_ID = 1


def make_payload(xyz, frame, sec, nanosec):
    return {
        'header': {'frame_id': frame, 'stamp': {'sec': sec, 'nanosec': nanosec}},
        'pose': {'position': dict(zip(('x', 'y', 'z'), xyz)),
                 'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}},
    }



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xyz', nargs=3, type=float, default=XYZ, metavar=('X', 'Y', 'Z'),
                        help='米制坐标；省略时使用脚本顶部XYZ')
    parser.add_argument('--topic', default=TOPIC)
    parser.add_argument('--frame', default=FRAME_ID)
    parser.add_argument('--rate', type=float, default=RATE_HZ)
    parser.add_argument('--class-id', type=int, choices=(1, 2), default=CLASS_ID)
    parser.add_argument('--class-topic', default='/vision/part_target')
    parser.add_argument('--check', action='store_true', help='仅检查并预览JSON，不启动ROS、不发布')
    args = parser.parse_args()
    if any(v is None or not math.isfinite(v) for v in args.xyz):
        parser.error('请先填写脚本顶部XYZ，或者传入 --xyz X Y Z（单位米）')
    if not math.isfinite(args.rate) or not 0 < args.rate <= 100:
        parser.error('--rate必须在(0,100]Hz内')
    if not args.frame or not args.topic.startswith('/'):
        parser.error('frame不能为空，topic必须以/开头')
    if args.check:
        print(json.dumps(make_payload(args.xyz, args.frame, 1, 0), ensure_ascii=False, allow_nan=False))
        print('仅格式预览；运行发布时每条消息使用当前ROS时间。')
        print(f'同时发布同帧类别 part{args.class_id} 到 {args.class_topic}')
        return

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import String
    rclpy.init(args=[])
    node = rclpy.create_node('manual_test_point_publisher')
    try:
        publisher = node.create_publisher(PoseStamped, args.topic, 10)
        class_publisher = node.create_publisher(String, args.class_topic, 10)

        def publish():
            stamp = node.get_clock().now().to_msg()
            msg = PoseStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = args.frame
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = args.xyz
            msg.pose.orientation.w = 1.0
            data = {'detected': True, 'class': f'part{args.class_id}', 'confidence': 1.0,
                    'frame_id': args.frame, 'position': dict(zip(('x','y','z'),args.xyz)),
                    'stamp': {'sec': stamp.sec, 'nanosec': stamp.nanosec}}
            class_publisher.publish(String(data=json.dumps(data, allow_nan=False)))
            publisher.publish(msg)

        timer = node.create_timer(1.0 / args.rate, publish)
        node.get_logger().info(
            f'测试点持续发布: {args.topic}; frame={args.frame}; XYZ(m)={args.xyz}; '
            f'{args.rate}Hz。Ctrl+C停止。修改XYZ后重启本脚本。')
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
